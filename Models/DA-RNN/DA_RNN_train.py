"""
    Training helper function
"""


from DA_RNN_model import DARNN

def train(model, X, y, target_length, criterion, optimizer, teacher_forcing = True):
    prediction = model(X, y, target_length, teacher_forcing = teacher_forcing)
    loss = criterion(prediction, y)
    loss.backward()
    optimizer.step()
    return loss