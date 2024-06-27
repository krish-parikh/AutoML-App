# Function to generate HTML and CSS code for a neural network diagram with Streamlit theme
def generate_nn_html_scrollable_compact(layers_config, max_neurons=10, max_display_layers=4):
    """
    Generates HTML and CSS code to visually represent a neural network with Streamlit theme.
    This version includes special input neurons with a tiny white circle inside and a scrollable view for many layers.
    
    Parameters:
        layers_config (list of dicts): The configuration of layers in the neural network.
        max_neurons (int): The maximum number of neurons to display per layer.
        max_display_layers (int): The maximum number of layers to display without scrolling.
        
    Returns:
        str: HTML and CSS code for the neural network diagram.
    """
    # Initialize HTML and CSS
    scrollable_style = "" if len(layers_config) <= max_display_layers else 'overflow-x: auto; white-space: nowrap;'
    html_code = f'<div style="display: flex; flex-direction: row; font-family: \'Integral CF\', sans-serif; color: #31333F; {scrollable_style}">'
    
    # Input layer
    input_neurons = min(layers_config[0]['neurons'], max_neurons)
    html_code += '<div class="layer">'
    for i in range(input_neurons):
        html_code += '<div class="neuron input-neuron"><div class="inner-circle"></div></div>'
    html_code += f'<div class="layer-label nowrap">Input Layer<br>({layers_config[0]["neurons"]})</div>'
    html_code += '</div>'
    
    # Hidden layers
    for i, layer in enumerate(layers_config[1:], 1):
        num_neurons = min(layer['neurons'], max_neurons)
        html_code += '<div class="layer">'
        for j in range(num_neurons):
            html_code += '<div class="neuron hidden-neuron"></div>'
        html_code += f'<div class="layer-label nowrap">Hidden {i}<br>({layer["neurons"]})</div>'
        html_code += '</div>'
    
    # Output layer
    html_code += '<div class="layer">'
    html_code += '<div class="neuron output-neuron"></div>'
    html_code += '<div class="layer-label nowrap">Output Layer<br>(1)</div>'
    html_code += '</div>'
    
    # Closing tag for the flex container
    html_code += '</div>'
    
    # CSS code for styling
    css_code = """
    <style>
        .layer {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0 20px;
        }
        .neuron {
            width: 0px;  /* Initially set to 0 for animation */
            height: 0px;  /* Initially set to 0 for animation */
            border-radius: 50%;
            margin: 5px;
            position: relative;
            animation: growNeuron 0.5s forwards;  /* Grow animation */
        }
        @keyframes growNeuron {
            to {
                width: 20px;
                height: 20px;
            }
        }
        @keyframes shrinkNeuron {
            from {
                width: 20px;
                height: 20px;
            }
            to {
                width: 0px;
                height: 0px;
            }
        }
        .input-neuron {
            background-color: #31333F;
        }
        .inner-circle {
            position: absolute;
            top: 50%;
            left: 50%;
            width: 10px;
            height: 10px;
            background-color: #FAFAFA;
            border-radius: 50%;
            transform: translate(-50%, -50%);
        }
        .hidden-neuron {
            background-color: #FF4B4B;
        }
        .output-neuron {
            background-color: #31333F;
        }
        .layer-label {
            margin-top: 10px;
            text-align: center;
            font-size: 14px;
            color: #31333F;  /* Streamlit theme color */
        }
        .nowrap {
            white-space: nowrap;
        }
    </style>
    """
    
    return css_code + html_code