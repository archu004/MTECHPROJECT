Setup the Python backend

Navigate to the server folder:

cd server


(Optional) Create a virtual environment:

python -m venv venv


Activate the virtual environment:

Windows:

venv\Scripts\activate


Linux/Mac:

source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Run the FastAPI server:

uvicorn main:app --reload


The server will start at http://127.0.0.1:8000.

3️⃣ Setup the React frontend

Open a new terminal and go to the client folder:

cd client


Install frontend dependencies:

npm install


Start the React development server:

npm start


The frontend will start at http://localhost:3000 and communicate with the backend.
