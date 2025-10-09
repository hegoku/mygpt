import mlx.core as mx
import torch
import MLXLlama
import mlx
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from mlx.utils import tree_flatten
import CosScheduler

def calc_loss_batch(model, input_batch, target_batch, ignore_index=-100):
    input_batch = mx.array(input_batch)
    target_batch = mx.array(target_batch)
    logits = model(input_batch)
    B, T, C = logits.shape # (batch_size, seq_len, vocab_size)
    logits = logits.reshape(B*T, C)
    target_batch = target_batch.reshape(B*T)
    mask = target_batch.flatten() != ignore_index
    loss = mlx.nn.losses.cross_entropy(logits, target_batch) * mask
    act = mask.sum()
    loss = loss.sum() / act
    return loss


def calc_loss_loader(data_loader, model, tokenizer, num_batches=None):
    total_loss = 0.
    # if len(data_loader) == 0:
    #     return float("nan")
    # elif num_batches is None:
    #     num_batches = len(data_loader)
    # else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        # num_batches = min(num_batches, len(data_loader))
    for i, batch in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(model, batch['input'], batch['target'], ignore_index=tokenizer.pad_token_id)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def train_model_simple(model, train_loader, val_loader, optimizer, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, write_log=False, save_dir=None, save_freq=1000):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    total_itr = len(train_loader)
    # scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00001)
    # scheduler = CosScheduler.MLXCosScheduler(optimizer, 0.00025, 0.00001, total_itr*num_epochs, 0.2)
    loss_and_grad = mlx.nn.value_and_grad(model, calc_loss_batch)

    accumulation_steps = 1
    mlx.core.eval(model.parameters()) # Create the model params (mlx is lazy evaluation)

    if write_log:
        run = wandb.init(
            # Set the wandb project where this run will be logged.
            project="mygpt",
            # Track hyperparameters and run metadata.
            config={
                # "learning_rate": scheduler.get_last_lr()[0],
                "epochs": num_epochs,
            },
        )

    # Main training loop
    for epoch in range(num_epochs):
        pbar = tqdm(total=len(train_loader))
        pbar.set_description_str(f"Ep {epoch+1}/{num_epochs} (Step {0:010d})")
        pbar.set_postfix_str(f"Loss 0 Val 0 lr: 0")
        model.train(True)  # Set model to training mode
        
        for b_idx, batch in enumerate(train_loader):
            loss, grads = loss_and_grad(model, batch['input'], batch['target'], tokenizer.pad_token_id)
            optimizer.update(model, grads)
            # mx.eval(model.parameters(), optimizer.state)
            mx.eval(model.parameters())
            # loss = loss.item()
            # loss = loss / accumulation_steps
            tokens_seen += batch['input'].numel()
            global_step += 1
            # scheduler.step(global_step)

            # if (b_idx + 1) % accumulation_steps == 0:
            #     optimizer.step() 
            #     optimizer.zero_grad()

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, val_loader, tokenizer, eval_iter)
                # train_losses.append(train_loss)
                # val_losses.append(val_loss)
                # track_tokens_seen.append(tokens_seen)
                if write_log:
                    run.log({"epoch": epoch, "loss":loss.item()*accumulation_steps, "val_loss":val_loss, "lr":optimizer.learning_rate})
                pbar.set_description_str(f"Ep {epoch+1}/{num_epochs} (Step {global_step:09d})")
                pbar.set_postfix_str(f"Loss {loss.item():.3f}  Val {val_loss:.3f}, lr {optimizer.learning_rate:.7f}")
                pbar.update(eval_freq)
                # print(f"Ep {epoch+1} (Step {global_step:06d}): "
                    #   f"Loss {loss:.3f}, Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, lr {scheduler.get_last_lr()[0]}")
            
            if save_dir!=None and global_step>0 and (global_step % save_freq ==0):
                model.save_weights(save_dir+'/mlx_model.npz')
                # flat_params = tree_flatten(model.parameters())
                # mx.savez("checkpoint/model.npz", **dict(flat_params))
                # torch.save({
                #     "model":model,
                #     "opt":optimizer,
                #     "scheduler":scheduler,
                #     "epoch":epoch,
                #     "trainer":train_loader,
                #     "val":val_loader,
                #     "num_epochs":num_epochs,
                #     "global_step":global_step
                # }, f"{save_dir}/model_{epoch+1}.chk")

        # if (b_idx + 1) % accumulation_steps != 0:
        #     optimizer.step()
        #     optimizer.zero_grad()

        # scheduler.step()
        pbar.close()

        if save_dir!=None:
            model.save_weights(save_dir+'/mlx_model.npz')
            # flat_params = tree_flatten(model.parameters())
            # mx.savez("checkpoint/model.npz", **dict(flat_params))
            # torch.save({
            #     "model":model,
            #     "opt":optimizer,
            #     "scheduler":scheduler,
            #     "epoch":epoch,
            #     "trainer":train_loader,
            #     "val":val_loader,
            #     "num_epochs":num_epochs,
            #     "global_step":global_step
            # }, f"{save_dir}/model_{epoch+1}.chk")

        # Print a sample text after each epoch
        pbar.write(generate_and_print_sample(
            model, tokenizer, start_context
        ))
    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, val_loader, tokenizer, eval_iter):
    return 0, 0
    model.train(False)
    train_loss = 0
    # train_loss = calc_loss_loader(train_loader, model, tokenizer, num_batches=eval_iter)
    val_loss = calc_loss_loader(val_loader, model, tokenizer, num_batches=eval_iter)
    model.train(True)
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, start_context):
    model.train(False)
    # context_size = model.pos_emb.weight.shape[0]
    # encoded = text_to_token_ids(start_context, tokenizer).to(device)
    encoded = tokenizer.encode(start_context)
    token_ids = MLXLlama.generate_text_batch(
        model=model, idx=mx.array(encoded.detach().clone().unsqueeze(0).numpy()),
        max_tokens=30, max_context=model.max_context
    )
    # decoded_text = token_ids_to_text(token_ids, tokenizer)
    decoded_text = tokenizer.decode(torch.tensor(memoryview(token_ids[0])))
    model.train(True)
    return decoded_text

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()