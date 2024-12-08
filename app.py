from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Access the posted data
        name = request.form['name']

        # Process the data (e.g., print it)
        print(f"Name: {name})

        # Return a response
        return f"Hello, {name}!"
    else:
        # Handle GET requests
        return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)
