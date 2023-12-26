from tqdm import tqdm
import torch


device = "cuda"

def train(dataset, data_loader, model, criterion, optimizer):
    model.train()
    num_batches = int(len(dataset) / data_loader.batch_size)

    tk0 = tqdm(data_loader, total=num_batches)
    
    for data in tk0:
        inputs = data["image"]
        targets = data["label"]
        
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        
        outputs = model(inputs)    
        loss = criterion(outputs, targets)
        loss.backward() 
        optimizer.step()
    tk0.close()

def evaluate(dataset, data_loader, model):
    model.eval()

    final_loss = 0.

    num_batches = int(len(dataset) / data_loader.batch_size)
    tk0 = tqdm(data_loader, total=num_batches)
    
    with torch.no_grad():
        for data in tk0:
            inputs = data["image"]
            targets = data["label"]
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            output = model(inputs)
            
            loss = criterion(output, targets)
            final_loss += loss.item()
    tk0.close()
    return  final_loss / num_batches