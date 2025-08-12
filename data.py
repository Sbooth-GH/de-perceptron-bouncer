import numpy as np
def get_training_data():
    features =  [
         "wears_i_love_javascript_tshirt",
        "is_carrying_menacing_rubber_duck",
        "tries_to_pay_with_pennies",
        "jumps_the_queue", ]
    
    dataset = [
    [1, 1, 1, 1],  
    [1, 1, 1, 0],          
    [1, 1, 0, 0],  
	[0, 1, 1, 0],  
	[0, 0, 0, 1],  
    [0, 1, 0, 0],  
	[0, 0, 0, 0],  
    ]

    targets = [0, 0, 0, 1, 1, 1, 1]  

    return {
        "features": np.array(features),
        "dataset": np.array(dataset),
        "targets": np.array(targets),
    }

