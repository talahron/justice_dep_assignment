import gradio as gr
import pickle
import pandas as pd

# Load the model at startup
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

FEATURE_COLUMNS = [f'feature_{i}' for i in range(10)]

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
with gr.Blocks(title="מודל תחזית תביעה", css="body {background-color: #f0f8ff; font-family: Arial, sans-serif;}") as demo:
    gr.Markdown("# מודל תחזית תביעה", elem_id="title")

    gr.Markdown("""
    <div style='text-align: center; color: #4a4a4a;'>
    העלה קובץ נתוני תביעה כדי לקבל תחזית
    </div>
    """, elem_id="description")

    file_input = gr.File(
        label="העלה קובץ CSV",
        file_types=[".csv"],
        elem_id="file_input"
    )
    prediction_output = gr.Textbox(
        label="תוצאת התחזית",
        max_lines=3,
        elem_id="prediction_output"
    )

    file_input.change(
        predict_single_row,
        inputs=[file_input],
        outputs=prediction_output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080, share=True)
