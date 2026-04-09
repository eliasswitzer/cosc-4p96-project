import numpy as np

best_architectures_multilabel = [
    {'num_hidden_layers': 4, 'hidden_layer_size': 158, 'learning_rate': np.float64(0.002345829390285611), 'momentum': np.float64(0.5458016213920623), 'batch_size': 189, 'weight_decay': np.float64(0.00042310646012446096), 'dropout_rate': np.float64(0.49038209919230774)},
    {'num_hidden_layers': 2, 'hidden_layer_size': 494, 'learning_rate': np.float64(0.0720876401520296), 'momentum': np.float64(0.2678506923588762), 'batch_size': 188, 'weight_decay': np.float64(0.003481152724366177), 'dropout_rate': np.float64(0.36792230072978865)},
    {'num_hidden_layers': 1, 'hidden_layer_size': 507, 'learning_rate': np.float64(0.07007243094258023), 'momentum': np.float64(0.2815590938874142), 'batch_size': 124, 'weight_decay': np.float64(0.003623917664421841), 'dropout_rate': np.float64(0.3520194747252782)}
]
best_architectures_multiclass = [
    {'num_hidden_layers': 4, 'hidden_layer_size': 375, 'learning_rate': np.float64(0.0061491327557080715), 'momentum': np.float64(0.7152189487445193), 'batch_size': 94, 'weight_decay': np.float64(0.00036178865562231413), 'dropout_rate': np.float64(0.11413161543947781)},
    {'num_hidden_layers': 4, 'hidden_layer_size': 255, 'learning_rate': np.float64(0.0039819634301220905), 'momentum': np.float64(0.3397462359893607), 'batch_size': 191, 'weight_decay': np.float64(0.0004385722446796244), 'dropout_rate': np.float64(0.029838948304784174)},
    {'num_hidden_layers': 3, 'hidden_layer_size': 76, 'learning_rate': np.float64(0.00828077392501765), 'momentum': np.float64(0.5970295271268181), 'batch_size': 147, 'weight_decay': np.float64(0.00034276383377430844), 'dropout_rate': np.float64(0.15206039451359205)},
    {'num_hidden_layers': 1, 'hidden_layer_size': 313, 'learning_rate': np.float64(0.003366724169904804), 'momentum': np.float64(0.7932681515803419), 'batch_size': 249, 'weight_decay': np.float64(0.00023434593921697671), 'dropout_rate': np.float64(0.10176387412120533)},
    {'num_hidden_layers': 2, 'hidden_layer_size': 325, 'learning_rate': np.float64(0.004433504616170434), 'momentum': np.float64(0.7775049978766315), 'batch_size': 203, 'weight_decay': np.float64(0.0002725926052826416), 'dropout_rate': np.float64(0.13823212757154835)},
    {'num_hidden_layers': 3, 'hidden_layer_size': 343, 'learning_rate': np.float64(0.0064135046161704345), 'momentum': np.float64(0.5795049978766316), 'batch_size': 155, 'weight_decay': np.float64(0.0004725926052826416), 'dropout_rate': np.float64(0.23823212757154835)}, #0.8618
    {'num_hidden_layers': 3, 'hidden_layer_size': 382, 'learning_rate': np.float64(0.008393504616170435), 'momentum': np.float64(0.38150499787663156), 'batch_size': 120, 'weight_decay': np.float64(0.0006725926052826416), 'dropout_rate': np.float64(0.33823212757154836)}, #.839
    {'num_hidden_layers': 2, 'hidden_layer_size': 289, 'learning_rate': np.float64(0.0023122658283643445), 'momentum': np.float64(0.998799028991683), 'batch_size': 264, 'weight_decay': np.float64(0.00025458847802880657), 'dropout_rate': np.float64(0.06607778747665126)} #.849 - (.75testing), 0.66 complex
]