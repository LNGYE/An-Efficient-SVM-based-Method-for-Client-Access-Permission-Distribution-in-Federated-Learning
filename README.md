# DASFAA2024
## An Efficient SVM-based Method for Client Access Permission Distribution in Federated Learning

### config.py:
1. Model selection, handling different datasets
model_name: str = 'femnist_cnn'
2. Global model update method
method: str = 'fedavg'
3. Data partitioning method
partition: str = "noniid-labeldir"
    
### main.py:
1. Verify whether the pre-order preparation is correct by checking the local accuracy and global accuracy of each local client, and save the accuracy
2. Randomly obtain (u, S) pairs
3. Test each pair and obtain the samples
4. Save samples of each dataset separately
5. Classify the sample and save it in the classification result
6. Training dataset construction, constructing 5 datasets respectively
7. Each classifier classifies and outputs the results
#### You can run main.py directly

### train.py:
1. Train the collection of clients
function train
2. Train a single client
function train_isolated

### std.py:
Calculate the mean and standard deviation of data

### clients.py:
Get client cluster

### getData.py:
Obtain the processed data of each dataset
