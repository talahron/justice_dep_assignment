import gradio as gr
import pickle
import pandas as pd

# Load the model at startup
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

FEATURE_COLUMNS = [
    'feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4',
    'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9',
    'feature_10', 'feature_11', 'feature_12', 'feature_13', 'feature_14',
    'feature_15', 'feature_16', 'feature_17', 'feature_18', 'feature_19'
]

def predict_single_row(file):
    """
    Make a prediction for a single row in the uploaded CSV file.
    """
    # Read the CSV file
    df = pd.read_csv(file)

    # Check if the file has at least one row
    if df.shape[0] < 1:
        return "הקובץ שהועלה הוא ריק - בבקשה העלה קובץ חדש"

    # Extract the first row of features
    features = df.loc[0, FEATURE_COLUMNS].values.reshape(1, -1)

    # Make the prediction
    prediction = model.predict(features)[0]

    if prediction == 0:
        answer = "נדחה"
    elif prediction == 1:
        answer = "פשרה"
    else:
        answer = "מאושר"

    # Create the result message
    result_message = f"המודל חוזה שהתביעה תסתיים בסטטוס: {answer}"
    return result_message

# Create Gradio interface
with gr.Blocks(title="מודל תחזית תביעה") as demo:
    gr.Markdown("# מודל תחזית תביעה")

    gr.Markdown("""
    העלה קובץ נתוני תביעה כדי לקבל תחזית
    """)

    file_input = gr.File(
        label="Upload CSV file",
        file_types=[".csv"]
    )
    prediction_output = gr.Textbox(
        label="Prediction Result",
        max_lines=3
    )

    file_input.change(
        predict_single_row,
        inputs=[file_input],
        outputs=prediction_output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)