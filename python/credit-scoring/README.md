## 🚀 Quick Start: Running the API

To run the Credit Scoring API locally, ensure you have **Python 3.11** installed.

**1. Navigate to the service directory:**

```bash
cd ./python/credit-scoring
```

**2. Install dependencies:**
Choose the appropriate requirements file based on your hardware:
```bash
# For standard CPU usage
pip install -r requirements.txt

# OR, if you have a compatible GPU
pip install -r requirements-gpu.txt
```
*(Note: Make sure `fastapi[standard]` and `dvc` are included in your requirements file).*

**3. Pull the model and data:**
Download the versioned models and required datasets from the DAGsHub remote storage:
```bash
dvc pull
```

**4. Start the FastAPI server:**
```bash
fastapi dev src/server/main.py
```

**5. Test the API:**
The service is now ready to be consumed! Open your browser and navigate to the auto-generated Swagger UI to test the endpoints:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**
