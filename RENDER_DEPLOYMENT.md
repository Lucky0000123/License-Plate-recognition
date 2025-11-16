# 🚀 Render Deployment Guide

## ✅ Fixed Issues

The deployment configuration has been updated to fix the Python version compatibility issue:

- ✅ Added `runtime.txt` specifying Python 3.11.0 (compatible with TensorFlow 2.15.0)
- ✅ Added `render.yaml` for automated deployment
- ✅ Added `gunicorn` to requirements.txt for production server
- ✅ Updated CORS configuration for production
- ✅ Configured dynamic port binding for Render

---

## 📋 Deployment Steps

### Step 1: Go to Render Dashboard
1. Visit https://dashboard.render.com
2. Sign in with your GitHub account (if not already signed in)

### Step 2: Create New Web Service
1. Click **"New +"** button in the top right
2. Select **"Web Service"**
3. Connect your repository: `Lucky0000123/License-Plate-recognition`
4. Click **"Connect"**

### Step 3: Configure the Service

Render should automatically detect the `render.yaml` file. If not, use these settings:

**Basic Settings:**
- **Name:** `license-plate-backend` (or your preferred name)
- **Region:** Choose closest to your location
- **Branch:** `main`
- **Runtime:** `Python 3`

**Build & Deploy:**
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn --bind 0.0.0.0:$PORT --timeout 120 backend.app:app`

**Plan:**
- Select **"Free"** plan

### Step 4: Deploy
1. Click **"Create Web Service"**
2. Wait for the build to complete (5-10 minutes for first deployment)
3. Watch the logs for any errors

### Step 5: Get Your URL
Once deployed, you'll see your service URL at the top:
- Example: `https://license-plate-backend.onrender.com`

### Step 6: Test the Deployment
1. Visit: `https://your-service-url.onrender.com/api/health`
2. You should see:
   ```json
   {
     "status": "healthy",
     "message": "License Plate Recognition API is running",
     "models_loaded": {
       "plate_detector": false,
       "char_recognizer": false
     }
   }
   ```

---

## ⚠️ Important Notes

### Free Tier Limitations:
- **Spin Down:** Service sleeps after 15 minutes of inactivity
- **Wake Up Time:** First request after sleep takes 30-60 seconds
- **Monthly Hours:** 750 hours/month free

### Model Files:
- Trained models (`.h5` files) are not included in the repository (they're in `.gitignore`)
- The API will work but use untrained models initially
- You'll need to train models or upload pre-trained ones

---

## 🔧 Troubleshooting

### Build Fails with TensorFlow Error
- ✅ **Fixed!** The `runtime.txt` now specifies Python 3.11.0

### Service Won't Start
- Check the logs in Render dashboard
- Verify the start command is correct
- Ensure all dependencies are in `requirements.txt`

### Health Check Fails
- The health check endpoint is `/api/health`
- Make sure the service is fully started before testing

---

## 📝 Next Steps

After deployment succeeds:

1. **Copy your service URL** (e.g., `https://license-plate-backend.onrender.com`)
2. **Share it with me** so I can:
   - Update the README with a live demo badge
   - Add the URL to GitHub repository's About section
   - Create a simple frontend deployment guide

---

## 🎯 Expected Result

Once deployed, your API will be accessible at:
- **Health Check:** `https://your-url.onrender.com/api/health`
- **Prediction API:** `https://your-url.onrender.com/api/predict`

The service will auto-deploy whenever you push to the `main` branch on GitHub!

