# Deploying Custom RAG Assistant to Render

This repository is fully configured for deployment on [Render](https://render.com) using Docker, Docker Compose, and Render Blueprints (`render.yaml`).

---

## 🚀 Option 1: 1-Click Render Blueprint (Recommended)

Render Blueprints allows you to launch both the **FastAPI Backend** and the **Streamlit Frontend** simultaneously.

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Add Docker and Render configuration"
   git push origin main
   ```
2. Open the [Render Dashboard](https://dashboard.render.com).
3. Click **New +** > **Blueprint**.
4. Connect your GitHub repository: `dhruvgupta1323/rag_pdf_project`.
5. Render will automatically parse [render.yaml](file:///render.yaml) and discover two web services:
   - **`rag-backend`** (FastAPI)
   - **`rag-frontend`** (Streamlit)
6. For each service, provide your `GEMINI_API_KEY` under the Environment Variables prompt.
7. Click **Apply**. Render will build and deploy both services!

---

## 🛠 Option 2: Deploy Individually as Render Web Services

If you only want to deploy one service (or prefer manual setup):

### Deploying the Streamlit Frontend:
1. On Render, click **New +** > **Web Service**.
2. Select your repository `dhruvgupta1323/rag_pdf_project`.
3. Configure the service:
   - **Name:** `rag-frontend`
   - **Runtime:** `Docker`
   - **Dockerfile Path:** `Dockerfile.frontend` (or `Dockerfile` with `SERVICE_TYPE=frontend`)
   - **Instance Type:** `Free`
4. Add Environment Variable:
   - `GEMINI_API_KEY` = `<your-google-gemini-api-key>`
5. Click **Create Web Service**.

### Deploying the FastAPI Backend:
1. On Render, click **New +** > **Web Service**.
2. Select your repository `dhruvgupta1323/rag_pdf_project`.
3. Configure the service:
   - **Name:** `rag-backend`
   - **Runtime:** `Docker`
   - **Dockerfile Path:** `Dockerfile.backend`
   - **Instance Type:** `Free`
4. Add Environment Variable:
   - `GEMINI_API_KEY` = `<your-google-gemini-api-key>`
5. Click **Create Web Service**.

---

## 🐳 Option 3: Run Locally with Docker Compose

To test the entire stack locally with Docker Compose:

1. Ensure `.env` exists and contains your `GEMINI_API_KEY`:
   ```bash
   GEMINI_API_KEY=your_key_here
   ```
2. Build and start both containers:
   ```bash
   docker compose up --build
   ```
3. Access your services:
   - **Streamlit Frontend:** [http://localhost:8501](http://localhost:8501)
   - **FastAPI Backend:** [http://localhost:8000](http://localhost:8000)
   - **Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
4. To stop:
   ```bash
   docker compose down
   ```

---

## ⚙️ Architecture & Memory Optimization

- **CPU-Only PyTorch:** The Dockerfiles are configured to pull lightweight CPU wheels for PyTorch rather than multi-gigabyte CUDA binaries, keeping build times fast and memory consumption low.
- **Port Handling:** Render dynamically assigns ports through the `$PORT` environment variable. The entry scripts automatically bind to `0.0.0.0:$PORT` with sensible fallbacks (8000/8501) for local execution.
