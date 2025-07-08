import math

class CosScheduler:
    def __init__(self, optimizer, max_lr, min_lr, total_steps, warm_up_ratio=0.0):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warm_up_ratio = warm_up_ratio
        self.total_steps = total_steps
        self.warmup_steps = total_steps * warm_up_ratio
        self.lr_inc = 0
        if self.warmup_steps!=0:
            self.lr_inc = (max_lr - min_lr) / self.warmup_steps
        self.last_lr = 0
        self.current_step = 0
        self.step(0)

    def step(self, step):
        self.current_step = step
        if step < self.warmup_steps:
            lr = self.min_lr + step * self.lr_inc
        else:
            lr = self.cos(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.last_lr = lr

    def cos(self, step):
        progress = ((step - self.warmup_steps) / 
                        (self.total_steps - self.warmup_steps))
        lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (
                1 + math.cos(math.pi * progress))
        if lr>self.max_lr:
            lr = self.max_lr
        return lr
        
    def get_last_lr(self):
        return self.last_lr
    
    def state_dict(self):
        return {
            "max_lr":self.max_lr,
            "min_lr":self.min_lr,
            "warm_up_ratio": self.warm_up_ratio,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "lr_inc": self.lr_inc,
            "last_lr": self.last_lr,
            "current_step":self.current_step
        }
    
    def load_state_dict(self, config):
        self.max_lr = config['max_lr']
        self.min_lr = config['min_lr']
        self.warm_up_ratio = config['warm_up_ratio']
        self.total_steps = config['total_steps']
        self.warmup_steps = config['warmup_steps']
        self.lr_inc = config['lr_inc']
        self.last_lr = config['last_lr']
        self.current_step = config['current_step']
        self.warmup_steps = self.total_steps * self.warm_up_ratio
        if self.warmup_steps!=0:
            self.lr_inc = (self.max_lr - self.min_lr) / self.warmup_steps
        self.step(self.current_step)

    def get_max_lr(self):
        return self.max_lr