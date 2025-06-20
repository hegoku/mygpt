import torch
import MyGPT
import MyLlama
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR

def calc_loss_batch(input_batch, target_batch, model, device, ignore_index=-100):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten(), ignore_index=ignore_index)
    return loss


def calc_loss_loader(data_loader, model, tokenizer, device, num_batches=None):
    total_loss = 0.
    # Reduce the number of batches to match the total number of batches in the data loader
    # if num_batches exceeds the number of batches in the data loader
    num = 0
    for i, batch in enumerate(data_loader):
        if i < num_batches:
            # input_batch = []
            # target_batch = []
            # for input_b in batch['input']:
            #     input_batch.append(tokenizer.encode(input_b).tolist())
            # for traget_b in batch['target']:
            #     target_batch.append(tokenizer.encode(traget_b).tolist())
            # input_batch = torch.tensor(input_batch, device=device)
            # target_batch = torch.tensor(target_batch, device=device)
            loss = calc_loss_batch(batch['input'], batch['target'], model, device, ignore_index=tokenizer.pad_token_id)
            total_loss += loss.item()
            num += 1
        else:
            break
    if num!=0:
        return total_loss / num
    return 0

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, write_log=False, save_dir=None, save_freq=1000):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00001)

    if write_log:
        run = wandb.init(
            # Set the wandb project where this run will be logged.
            project="mygpt",
            # Track hyperparameters and run metadata.
            config={
                "learning_rate": scheduler.get_last_lr()[0],
                "epochs": num_epochs,
            },
        )

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        # train_loader.set_epoch(epoch)
        
        for b_i, batch in enumerate(train_loader):
            # input_batch = []
            # target_batch = []
            # for input_b in batch['input']:
                # input_batch.append(tokenizer.encode(input_b).tolist())
            # for traget_b in batch['target']:
                # target_batch.append(tokenizer.encode(traget_b).tolist())
            # input_batch = torch.tensor(input_batch, device=device)
            # target_batch = torch.tensor(target_batch, device=device)
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(batch['input'], batch['target'], model, device, ignore_index=tokenizer.pad_token_id)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += batch['input'].numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, tokenizer, device, eval_iter)
                # train_losses.append(train_loss)
                # val_losses.append(val_loss)
                # track_tokens_seen.append(tokens_seen)
                if write_log:
                    # run.log({"epoch": epoch, "loss":loss, "train_loss": train_loss, "val_loss":val_loss, "lr":scheduler.get_last_lr()[0]})
                    run.log({"epoch": epoch, "loss":loss, "val_loss":val_loss, "lr":scheduler.get_last_lr()[0]})
                # print(f"Ep {epoch+1} (Step {global_step:09d}): "
                    #   f"Loss {loss:.3f}, Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, lr {scheduler.get_last_lr()[0]}")
                print(f"Ep {epoch+1} (Step {global_step:09d}): "
                      f"Loss {loss:.3f}, Val loss {val_loss:.3f}, lr {scheduler.get_last_lr()[0]}")
                
            if save_dir!=None and global_step>0 and (global_step % save_freq ==0):
                torch.save({
                    "model":model,
                    "opt":optimizer.state_dict(),
                    "scheduler":scheduler.state_dict(),
                    "epoch":epoch,
                    "trainer":train_loader.state_dict(),
                    "val":val_loader.state_dict(),
                    "num_epochs":num_epochs,
                    "global_step":global_step
                }, f"{save_dir}/model_{epoch+1}.chk")
                print("Saved !!!")

        scheduler.step()

        torch.save({
            "model":model,
            "opt":optimizer.state_dict(),
            "scheduler":scheduler.state_dict(),
            "epoch":epoch,
            "trainer":train_loader.state_dict(),
            "val":val_loader.state_dict(),
            "num_epochs":num_epochs,
            "global_step":global_step
        }, f"{save_dir}/model_{epoch+1}.chk")
        print("Saved !!!")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, tokenizer, device, eval_iter):
    model.eval()
    train_loss = 0
    with torch.no_grad():
        # train_loss = calc_loss_loader(train_loader, model, tokenizer, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, tokenizer, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    # context_size = model.pos_emb.weight.shape[0]
    # encoded = text_to_token_ids(start_context, tokenizer).to(device)
    encoded = tokenizer.encode(start_context).to(device)
    with torch.no_grad():
        token_ids = MyLlama.generate_text(
            model=model, idx=encoded.detach().clone().unsqueeze(0),
            max_tokens=30, max_context=model.max_context
        )
    # decoded_text = token_ids_to_text(token_ids, tokenizer)
    decoded_text = tokenizer.decode(token_ids[0])
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

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