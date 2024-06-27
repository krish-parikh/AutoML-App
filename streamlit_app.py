import streamlit as st
from utilities import *
from upload import *
from profiling import *
from preprocessing import *
from model import *
from html_design import *
from visualisation import *
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit.components.v1 as components
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import tensorflow as tf
import time
import base64
import plotly.figure_factory as ff

with st.sidebar:
    st.image("Images/inno2-2.png")
    st.title("DragAI")
    mode = toggle_modes()
    choice = st.radio("Navigation", ["Upload", "Profiling", "Preprocessing", "Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")

    if is_new_dataset_present():
        if st.button("Reset Changes", key="reset_changes"):
            os.remove("dataset_updated.csv")
            st.success("Changes reset successfully!")
    
    # Button to delete all uploaded files
    if st.button('Delete All Uploaded Files'):
        delete_all_uploaded_files()

if choice == "Upload":
    st.title("Upload Selection")
    st.info("This section allows you to upload your data.")
    display_uploaded_dataset()
    st.subheader("Upload Options")
    upload_choice = st.radio("Choose how to upload your dataset:", ["Upload File", "Kaggle API"])
    
    if upload_choice == "Upload File":
        handle_file_upload()
    elif upload_choice == "Kaggle API":
        handle_kaggle_api_upload()

elif choice == "Profiling":
    profile_data()


elif choice == "Preprocessing":
    st.title("Preprocessing Selection")
    st.info("This section allows you to preprocess your data.")
    
    if is_dataset_present():
        df = dataset_selection()
        st.subheader("Current Dataset")
        st.dataframe(df)

        data_structure_choice = st.radio("Choose a data structure:", ["Structured", "Unstructured", "Time Series", "Text", "Image"])

        if data_structure_choice == "Structured":
            preprocessing_choice = st.selectbox("Choose a preprocessing technique:", ["Data Cleaning", "Data Transformation", "Data Reduction", 
                                                                            "Data Discretization", "Feature Engineering", "Data Intergration",
                                                                            "Encoding"])

            if preprocessing_choice == "Data Cleaning":
                cleaning_choice = st.multiselect("Choose a cleaning technique:", ["Missing Values", "Outlier Detection", "Drop Column or Row"])

                if "Missing Values" in cleaning_choice:
                    df, comparison_df = handle_missing_values(df)

                    if comparison_df is not None:
                        st.dataframe(comparison_df)

                    else:
                        st.success("No missing values found!")
                
                if "Drop Column or Row" in cleaning_choice:
                    if st.checkbox('Drop Column'):
                        df = drop_column(df)
                        st.write(df)

                    if st.checkbox('Drop Row'):
                        df = drop_rows(df)
                        st.write(df)

                if "Outlier Detection" in cleaning_choice:

                    numeric_cols = numeric_columns(df)
                    string_cols = string_columns(df)

                    with st.expander("Outlier Detection for Numeric Columns"):
                        method = st.selectbox("Choose an outlier detection method", ["IQR", "Z-Score"])
                        selected_col = st.selectbox("Select column", numeric_cols)
                        
                        if st.button("Detect Outliers", key="detect_outliers_numeric"):
                            if method == "IQR":
                                outliers = iqr_outliers(df[selected_col])
                                fig = display_iqr(df, selected_col)
                            elif method == "Z-Score":
                                z_scores = zscore(df[selected_col], nan_policy='omit')
                                outliers = zscore_outliers(df[selected_col])
                                fig = display_zscore(z_scores, selected_col)
                                st.write(f"Mean: {df[selected_col].mean():.2f}, Standard Deviation: {df[selected_col].std():.2f}")

                            numeric_outlier_info(outliers, selected_col)
                            st.plotly_chart(fig)
                    
                    with st.expander("Outlier Detection for Categorical Columns"):
                        selected_col = st.selectbox("Select column", string_cols)
                        threshold = st.slider("Frequency Threshold", 1, 100, 5)  # Adjust max value as needed
                        
                        if st.button("Detect Outliers", key="detect_outliers_categorical"):
                            outliers_mask, outliers = categorical_outliers(df[selected_col], threshold=threshold)
                            fig = display_categorical(df, selected_col)

                            categorical_outlier_info(outliers, outliers_mask, selected_col)
                            st.plotly_chart(fig)

                save_dataset(df)
                
            elif preprocessing_choice == "Data Transformation":
                numeric_cols = numeric_columns(df)

                if not numeric_cols:
                    st.success("No numeric columns found! Please select another preprocessing technique.")
                else:
                    transformation_choice = st.selectbox("Choose a transformation technique:", ["Standardization", "Normalization"])
                    selected_columns = st.multiselect("Select columns to encode (default is all numeric columns):", options=numeric_cols, default=numeric_cols)
                    if selected_columns:
                        if transformation_choice == "Standardization":
                            if st.checkbox("Perform Standardization"):
                                df, success = standardize_data(df, selected_columns)
                                if success:
                                    st.dataframe(df)
                                    st.success("Standardization performed successfully!")
                        elif transformation_choice == "Normalization":
                            if st.checkbox("Perform Normalization"):
                                df, success = normalize_data(df, selected_columns)
                                if success:
                                    st.dataframe(df)
                                    st.success("Normalization performed successfully!")
                    else:
                        st.warning("You haven't selected any columns. It's recommended to transform all numeric columns.")
                    
                    save_dataset(df)

            
            elif preprocessing_choice == "Encoding":
                categorical_cols = categorical_columns(df)

                if not categorical_cols:
                    st.success("No categories that need to be encoded! Please select another preprocessing technique.")
                
                else:
                        
                    encoding_choice = st.selectbox("Choose an encoding technique:", ["Label Encoding", "One-Hot Encoding"])
                    selected_columns = st.multiselect("Select columns to encode (default is all categorical columns):", options=categorical_cols, default=categorical_cols)

                    if not selected_columns:
                        st.warning("You haven't selected any columns. It's recommended to encode all categorical columns as most ML algorithms require numerical input features.")

                    if "One-Hot Encoding" in encoding_choice:
                        st.write("One-Hot Encoding is selected when the categories are not ordered in any way.")
                        if st.checkbox("Perform One-Hot Encoding"):
                            df = one_hot_encode(df, selected_columns)
                            st.dataframe(df)
                            st.success("One-Hot Encoding performed successfully!")

                    if "Label Encoding" in encoding_choice:
                        st.write("Label Encoding is selected when categories are ordered in some way.")
                        if st.checkbox("Perform Label Encoding"):
                            df = label_encode(df, selected_columns)
                            st.dataframe(df)
                            st.success("Label Encoding performed successfully!")
                    
                    save_dataset(df)

    else:
        st.error("No dataset found. Please upload a file in the 'Upload' section.")
                


elif choice == "Modelling":
    st.title("Modelling Selection")
    st.info("This section allows you to build a model using your data.")

    if is_dataset_present():
        df = dataset_selection()
        target = st.selectbox("Select the target column:", df.columns)
        X, y = feature_target_split(df, target)
        
        num_features = X.shape[1]

        split_choice = st.selectbox("Choose a split technique:", ["Train-Test Split", "No Split"])
        
        if split_choice == "Train-Test Split":
            # Setup test size
            test_size = st.slider("Test Size %", 0.0, 1.0, 0.2)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        elif split_choice == "No Split":
            X_train, y_train = X, y
            X_test, y_test = None, None
        
        # Setup task
        task_choice = st.selectbox("Choose a task:", ["Classification", "Regression"])
                
        st.subheader("Neural Network Configuration")

        # Number of Layers
        num_layers = st.number_input("Number of Hidden Layers", min_value=1, max_value=5, value=1, step=1, key="num_layers")
        # Configuration for each layer
        layers_config = [{'neurons': num_features, 'activation': 'relu'}]  # Assuming 20 input features for simplicity

        for i in range(num_layers):
            with st.expander(f"Layer {i+1}"):
                neurons = st.number_input(f"Number of Neurons in Layer {i+1}", min_value=5, max_value=128, value=5, step=1)
                activation = st.selectbox(f"Activation Function for Layer {i+1}", ['relu', 'sigmoid', 'tanh', 'linear'])
                layers_config.append({'neurons': neurons, 'activation': activation})



        # Generate and display the neural network HTML
        nn_html = generate_nn_html_scrollable_compact(layers_config)
        components.html(nn_html, height=400)

        # Learning Rate
        learning_rate = st.number_input("Learning Rate", min_value=0.001, max_value=0.1, value=0.01, step=0.001)

        if task_choice == "Classification":
            # Setup loss function
            selected_loss = st.selectbox("Loss Function", ['Binary Cross-Entropy (Log Loss)', 'Hinge Loss', 'Squared Hinge Loss', 'Focal Loss', 'Kullback-Leibler Divergence'])

            if selected_loss == 'Binary Cross-Entropy (Log Loss)':
                st.info("This is the go-to for most binary classification problems. It works well when your classes are balanced, meaning there's roughly the same number of yes's and no's in your dataset.")
                loss = 'binary_crossentropy'
            elif selected_loss == 'Hinge Loss':
                st.info("When you want to maximize the margin between decision boundaries. It gives more freedom to your model to not be 'perfect,' but penalizes it if it's way off.")
                loss = 'hinge'
            elif selected_loss == 'Squared Hinge Loss':
                st.info("Similar to Hinge Loss, but penalizes outliers more. This is when you really want your model to get it right") 
                loss = 'squared_hinge'
            elif selected_loss == 'Focal Loss':
                st.info("When you have imbalanced classes. This loss function makes sure the model focuses on the minority class.")
                loss = 'focal_loss'
            elif selected_loss == 'Kullback-Leibler Divergence':
                st.info("When you're comparing two probability distributions. It measures how one distribution diverges from the second one.")
                loss = 'kullback_leibler_divergence'

            # Setup the model
            confirm_box = st.checkbox("Confirm and Build Model", key="confirm_classification", value=False)

            if confirm_box:
                # Build the model
                model = tf.keras.models.Sequential()
                model.add(tf.keras.layers.Input(shape=(num_features,)))

                for idx, layer in enumerate(layers_config):
                    layer_name = f"hidden_layer_{idx+1}"
                    model.add(tf.keras.layers.Dense(layer['neurons'], activation=layer['activation'], name=layer_name))
            
                model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss, metrics=['accuracy'])

                st.success("Model built successfully!")

                epochs = st.number_input("Epochs", min_value=1, max_value=150, value=3*num_features, step=1, key="epochs")

                train_model_button = st.button("Train Model", key="train_model_regression")

                if train_model_button:
                    progress_bar = st.progress(0)
                    for i in range(100):
                        # Update progress bar
                        time.sleep(0.1)
                        progress_bar.progress(i + 1)

                    # Train the model (assuming you have X_train and y_train)
                    model.fit(X_train, y_train, epochs=epochs, verbose=0)  # You can set epochs and other params

                    st.success("Model trained successfully! 🎉")

                    model_path = "trained_model.h5"
                    model.save(model_path)
        
                    # Create a link to download the model file
                    st.write("Download the trained model:")
                    with open(model_path, "rb") as f:
                        bytes = f.read()
                        b64 = base64.b64encode(bytes).decode("utf-8")
                        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{model_path}">Click Here to Download</a>'
                        st.markdown(href, unsafe_allow_html=True)

                    # Evaluate the model
                    st.subheader("Model Evaluation Metrics")
                    predictions_classification = model.predict(X_test)
                    predictions_classification = np.round(predictions_classification)  # Assuming binary classification

                    # Display classification metrics
                    st.write(f"Accuracy: {metrics.accuracy_score(y_test, predictions_classification)}")
                    st.write(f"Precision: {metrics.precision_score(y_test, predictions_classification)}")
                    st.write(f"Recall: {metrics.recall_score(y_test, predictions_classification)}")
                    st.write(f"F1 Score: {metrics.f1_score(y_test, predictions_classification)}")

                    # Generate the confusion matrix
                    cm = metrics.confusion_matrix(y_test, predictions_classification)

                    # Create the Plotly Figure
                    labels = ['True Neg','False Pos','False Neg','True Pos']
                    labels = np.asarray(labels).reshape(2,2)
                    fig = ff.create_annotated_heatmap(cm, annotation_text=labels, colorscale='Blues')
                    fig.update_layout(title='Confusion Matrix',
                                    xaxis=dict(title='Predicted label'),
                                    yaxis=dict(title='True label'))

                    # Show the plot in Streamlit
                    st.plotly_chart(fig)



        if task_choice == "Regression":
            # Setup loss function
            selected_loss = st.selectbox("Loss Function", ['Mean Squared Error', 'Mean Absolute Error'])

            if selected_loss == 'Mean Squared Error':
                st.info("Use this when you want to give larger errors more weight. It's the 'tough love' approach to training a model.")
                loss = 'mean_squared_error'
            elif selected_loss == 'Mean Absolute Error':
                st.info("Mean Absolute Error treats all errors the same, whether they're big or small. If you don't want your model to freak out about large errors, this is a good option.")
                loss = 'mean_absolute_error'

            # Setup the model
            confirm_box = st.checkbox("Confirm and Build Model", key="confirm_regression", value=False)

            if confirm_box:
                # Build the model
                model = tf.keras.models.Sequential()
                model.add(tf.keras.layers.Input(shape=(num_features,)))

                for idx, layer in enumerate(layers_config):
                    layer_name = f"hidden_layer_{idx+1}"
                    model.add(tf.keras.layers.Dense(layer['neurons'], activation=layer['activation'], name=layer_name))
            
                model.add(tf.keras.layers.Dense(1, activation='linear'))

                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss, metrics=['mean_squared_error'])

                st.success("Model built successfully!")

                epochs = st.number_input("Epochs", min_value=1, max_value=150, value=3*num_features, step=1, key="epochs")

                train_model_button = st.button("Train Model", key="train_model_regression")

                if train_model_button:
                    progress_bar = st.progress(0)
                    for i in range(100):
                        # Update progress bar
                        time.sleep(0.1)
                        progress_bar.progress(i + 1)

                    # Train the model (assuming you have X_train and y_train)
                    model.fit(X_train, y_train, epochs=epochs, verbose=0)  # You can set epochs and other params

                    st.success("Model trained successfully! 🎉")

                    model_path = "trained_model.h5"
                    model.save(model_path)
        
                    # Create a link to download the model file
                    st.write("Download the trained model:")
                    with open(model_path, "rb") as f:
                        bytes = f.read()
                        b64 = base64.b64encode(bytes).decode("utf-8")
                        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{model_path}">Click Here to Download</a>'
                        st.markdown(href, unsafe_allow_html=True)

                    # Evaluate the model
                    st.subheader("Model Evaluation Metrics")
                    predictions_regression = model.predict(X_test)
                    st.write(f"Mean Squared Error: {metrics.mean_squared_error(y_test, predictions_regression)}")
                    st.write(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, predictions_regression)}")
                    st.write(f"R2 Score: {metrics.r2_score(y_test, predictions_regression)}")

                    y_test = np.array(y_test).reshape(-1)
                    predictions_regression = np.array(predictions_regression).reshape(-1)

                    # Generate the scatter plot
                    fig = px.scatter(x=y_test, y=predictions_regression, labels={'x': 'True Labels', 'y': 'Predictions'})

                    # Add a line for perfect predictions (optional)
                    fig.add_shape(
                        type="line", line=dict(dash='dash'),
                        x0=y_test.min(), y0=y_test.min(),
                        x1=y_test.max(), y1=y_test.max()
                    )

                    # Update the layout and show the plot
                    fig.update_layout(title='True Labels vs Predictions',
                                    xaxis_title='True Labels',
                                    yaxis_title='Predictions')

                    # To display the plot in your Streamlit app
                    st.plotly_chart(fig)

    else:
        st.error("No dataset found. Please upload a file in the 'Upload' section.")


elif choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")

        