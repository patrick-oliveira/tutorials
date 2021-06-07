import sys
import time
import logging

def log_training(log_path: str, log_name: str = 'default_log'):
    logging.basicConfig(stream = sys.stdout, 
                        level = logging.INFO, 
                        format = '%(message)s')
    logger = logging.getLogger('Training Logger')
    
    timestamp = time.strftime('%b-%d-%Y', time.localtime())
    handler = logging.FileHandler(filename = log_path + '/' + log_name + ' - ' + timestamp + '.log',
                                  mode = 'w')
    logger.addHandler(handler)
    
    try:
        yield logger
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
            
class Trainer:
    def __init__(self, problem_type: str = None,
                       training_routine,
                       validation_routine):
        '''
        Args:
           problem_type (str): It must be either "regression" or "classification".
        '''
        
        self.training_function = {'regression': train_regression,
                                  'classification': train_classification}[problem_type]
        self.training_routine  = training_routine
        self.validation_routine = validation_routine
        
    def train(self, model, dataloaders, criterion, optimizer, scheduler = None, num_epochs = 10):
        path = ''
        log_name = ''
        
        with log_training(path, log_name) as logger:
            model, training_info = self.training_function(model, dataloaders, criterion, optimizer, scheduler, num_epochs, logger, self.training_routine, self.validation_routine)
       
        return model, training_info
        
        
def train_classification(model, dataloaders, criterion, optimizer, scheduler = None, num_epochs = 10, logger = None,
                         training_routine = None, validation_routine = None):
    
    assert training_routine != None and validation_routine != None, 'Must provide training and validation routines.'
    
    header = "{:7}  {:10}  {:6}  {:8}\n".format("Epoch", "Stage", "Loss", "Accuracy")
    logger.info(header)
        
    best_model = copy.deepcopy(model.state_dict())
    training_info = {
        'Best Accuracy' : .0,
        'Best Loss' : np.inf,
        'Training Stats' : {'Accuracy' : [], 'Loss': []},
        'Validation Stats' : {'Accuracy' : [], 'Loss': []}
    }

    since = time.time()
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        cumulative_loss = .0
        cumulative_hits = 0
        for inputs, labels in dataloaders['train']:
            c_loss_temp, c_hits_temp = training_routine(model, inputs, labels, criterion, optimizer, scheduler)
            cumulative_loss += c_loss_temp
            cumulative_hits += c_hits_temp
            
        training_accuracy = cumulative_hits / len(dataloaders['train'].dataset)
        training_loss     = cumulative_loss / len(dataloaders['train'].dataset)
        training_info['Training Stats']['Accuracy'].append(training_accuracy)
        training_info['Training Stats']['Loss'].append(training_loss)
        
        
        
        # Validation Phase
        model.eval()
        cumulative_loss = .0
        cumulative_hits = 0
        for inputs, labels in dataloaders['val']:
            c_loss_temp, c_hits_temp = validation_routine(model, inputs, labels, criterion)
            cumulative_loss += c_loss_temp
            cumulative_hits += c_hits_temp
            
        validation_accuracy = cumulative_hits / len(dataloaders['val'].dataset)
        validation_loss     = cumulative_loss / len(dataloaders['val'].dataset)
        training_info['Validation Stats']['Accuracy'].append(validation_accuracy)
        training_info['Validation Stats']['Loss'].append(validation_loss)
        
        
        
        # Model evaluation and logging
        if training_info['Best Accuracy'] < validation_accuracy:
            training_info['Best Accuracy'] = validation_accuracy
            training_info['Best Loss']     = validation_loss
            best_model = copy.deepcopy(model.state_dict())
            
        training_log = "{:7}  {:10}  {:<6.2f}  {:<8.2f}".format("{}/{}".format(epoch + 1, num_epochs), "Training", training_loss, training_accuracy)
        validation_log = "{:7}  {:10}  {:<6.2f}  {:<8.2f}".format(" ", "Validation", validation_loss, validation_accuracy)
        logger.info(training_log); logger.info(validation_log)
            
    conclusion_log = 'Training complete in {:.0f}m {:.0f}s \n \
                      Best Validation Accuracy: {:.2f} \n \
                      Best Validation Loss: {:.2f}'.format(time_elapsed // 60, time_elapsed % 60, training_info['Best Accuracy'], training_info['Best Loss'])
    logger.info(conclusion_log)
    model.load_state_dict(best_model)
    return model, training_info

def train_regression(model, dataloaders, criterion, optimizer, scheduler = None, num_epochs = 10, logger = None,
                     training_routine = None, validation_routine = None):
    assert training_routine != None and validation_routine != None, 'Must provide training and validation routines.'
    
    header = "{:7}  {:10}  {:6}  {:8}\n".format("Epoch", "Stage", "Loss")
    logger.info(header)
        
    best_model = copy.deepcopy(model.state_dict())
    training_info = {
        'Best Loss' : np.inf,
        'Training Loss' : [],
        'Validation Loss' : []
    }

    since = time.time()
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        cumulative_loss = .0
        for inputs, labels in dataloaders['train']:
            c_loss_temp = training_routine(model, inputs, labels, criterion, optimizer, scheduler)
            cumulative_loss += c_loss_temp
            
        training_loss     = cumulative_loss / len(dataloaders['train'].dataset)
        training_info['Training Loss'].append(training_loss)
        
        
        
        # Validation Phase
        model.eval()
        cumulative_loss = .0
        for inputs, labels in dataloaders['val']:
            c_loss_temp = validation_routine(model, inputs, labels, criterion)
            cumulative_loss += c_loss_temp
            
        validation_loss     = cumulative_loss / len(dataloaders['val'].dataset)
        training_info['Validation Loss'].append(validation_loss)
        
        
        
        # Model evaluation and logging
        if training_info['Best Loss'] > validation_loss:
            training_info['Best Loss']     = validation_loss
            best_model = copy.deepcopy(model.state_dict())
            
        training_log = "{:7}  {:10}  {:<6.2f}".format("{}/{}".format(epoch + 1, num_epochs), "Training", training_loss)
        validation_log = "{:7}  {:10}  {:<6.2f}".format(" ", "Validation", validation_loss)
        logger.info(training_log); logger.info(validation_log)
            
    conclusion_log = 'Training complete in {:.0f}m {:.0f}s \n \
                      Best Validation Loss: {:.2f}'.format(time_elapsed // 60, time_elapsed % 60, training_info['Best Accuracy'], training_info['Best Loss'])
    logger.info(conclusion_log)
    model.load_state_dict(best_model)
    return model, training_info