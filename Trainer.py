import torch
import MyGPT
import MyLlama
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, write_log=False, save_dir=None, save_freq=1000):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001)

    accumulation_steps = 4

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
        
        for b_idx, (input_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # loss = loss / accumulation_steps
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # if (b_idx + 1) % accumulation_steps == 0:
                # optimizer.step()
                # optimizer.zero_grad()

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                # train_losses.append(train_loss)
                # val_losses.append(val_loss)
                # track_tokens_seen.append(tokens_seen)
                if write_log:
                    run.log({"epoch": epoch, "loss":loss, "train_loss": train_loss, "val_loss":val_loss, "lr":scheduler.get_last_lr()[0]})
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Loss {loss:.3f}, Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, lr {scheduler.get_last_lr()[0]}")
            
            if save_dir!=None and global_step>0 and (global_step % save_freq ==0):
                torch.save({
                    "model":model,
                    "opt":optimizer,
                    "scheduler":scheduler,
                    "epoch":epoch,
                    "trainer":train_loader,
                    "val":val_loader,
                    "num_epochs":num_epochs,
                    "global_step":global_step
                }, f"{save_dir}/model_{epoch+1}.chk")

        scheduler.step()

        torch.save({
            "model":model,
            "opt":optimizer,
            "scheduler":scheduler,
            "epoch":epoch,
            "trainer":train_loader,
            "val":val_loader,
            "num_epochs":num_epochs,
            "global_step":global_step
        }, f"{save_dir}/model_{epoch+1}.chk")

        # if (b_idx + 1) % accumulation_steps != 0:
        #     optimizer.step()
        #     optimizer.zero_grad()

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
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