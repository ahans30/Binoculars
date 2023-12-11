from demo.demo import app

if __name__ == "__main__":
    # Launch the Gradio interface
    app.launch(show_api=False, debug=True, share=True)
