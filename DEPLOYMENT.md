# Deployment Guide

This guide covers deploying your License Plate Recognition system to various platforms.

## Table of Contents

- [Local Development](#local-development)
- [GitHub Setup](#github-setup)
- [Heroku Deployment](#heroku-deployment)
- [Docker Deployment](#docker-deployment)
- [AWS Deployment](#aws-deployment)

## Local Development

For local development, follow the QUICK_START.md guide.

## GitHub Setup

### 1. Initialize Git Repository

```bash
cd /path/to/FINAL_PROJECT
git init
git add .
git commit -m "Initial commit: License Plate Recognition System"
```

### 2. Create GitHub Repository

1. Go to https://github.com/new
2. Create a new repository (e.g., `license-plate-recognition`)
3. Don't initialize with README (we already have one)

### 3. Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/license-plate-recognition.git
git branch -M main
git push -u origin main
```

### 4. GitHub Actions

The project includes a CI/CD workflow in `.github/workflows/main.yml` that will:
- Run tests on every push
- Check code quality with linting
- Build the frontend
- Generate coverage reports

## Heroku Deployment

### Backend (Flask API)

1. **Create Procfile** (already included):

```
web: cd backend && gunicorn app:app
```

2. **Install Heroku CLI** and login:

```bash
heroku login
```

3. **Create Heroku app**:

```bash
heroku create your-plate-recognition-api
```

4. **Add buildpacks**:

```bash
heroku buildpacks:add heroku/python
```

5. **Set environment variables**:

```bash
heroku config:set FLASK_ENV=production
```

6. **Deploy**:

```bash
git push heroku main
```

### Frontend (React)

Deploy to **Netlify** or **Vercel**:

**Netlify:**
```bash
cd frontend
npm run build
netlify deploy --prod --dir=build
```

**Vercel:**
```bash
cd frontend
vercel --prod
```

Update the API endpoint in `frontend/package.json` to your Heroku backend URL.

## Docker Deployment

### 1. Create Dockerfile for Backend

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY data/ ./data/

WORKDIR /app/backend

EXPOSE 5000

CMD ["python", "app.py"]
```

### 2. Create Dockerfile for Frontend

```dockerfile
FROM node:18-alpine as build

WORKDIR /app

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

### 3. Create docker-compose.yml

```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./backend/saved_models:/app/backend/saved_models

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "80:80"
    depends_on:
      - backend
```

### 4. Run with Docker

```bash
docker-compose up -d
```

## AWS Deployment

### Option 1: AWS Elastic Beanstalk

1. **Install EB CLI**:

```bash
pip install awsebcli
```

2. **Initialize EB**:

```bash
eb init -p python-3.10 license-plate-recognition
```

3. **Create environment**:

```bash
eb create production-env
```

4. **Deploy**:

```bash
eb deploy
```

### Option 2: AWS EC2

1. **Launch EC2 instance** (Ubuntu 22.04)

2. **SSH into instance**:

```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

3. **Setup environment**:

```bash
sudo apt update
sudo apt install python3-pip python3-venv nodejs npm nginx -y

git clone https://github.com/YOUR_USERNAME/license-plate-recognition.git
cd license-plate-recognition

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cd frontend
npm install
npm run build
```

4. **Configure Nginx**:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        root /path/to/frontend/build;
        try_files $uri /index.html;
    }

    location /api {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

5. **Setup systemd service**:

```ini
[Unit]
Description=License Plate Recognition API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/license-plate-recognition/backend
Environment="PATH=/home/ubuntu/license-plate-recognition/venv/bin"
ExecStart=/home/ubuntu/license-plate-recognition/venv/bin/python app.py

[Install]
WantedBy=multi-user.target
```

6. **Start services**:

```bash
sudo systemctl start nginx
sudo systemctl enable nginx
sudo systemctl start plate-recognition
sudo systemctl enable plate-recognition
```

## Environment Variables

Always set these environment variables in production:

```bash
# Backend
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=your-secret-key-here

# Optional
DATABASE_URL=your-database-url
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
```

## Security Considerations

1. **Never commit sensitive data**:
   - Add `.env` to `.gitignore`
   - Use environment variables for secrets
   - Don't commit model weights to git (use Git LFS or cloud storage)

2. **API Security**:
   - Add rate limiting
   - Implement API key authentication
   - Use HTTPS in production
   - Validate file uploads

3. **CORS Configuration**:
   - Restrict CORS to your frontend domain in production
   - Update `backend/app.py` CORS settings

## Performance Optimization

1. **Model Loading**:
   - Load models once at startup
   - Use model caching
   - Consider model quantization for faster inference

2. **Image Processing**:
   - Resize images before sending to API
   - Use image compression
   - Implement client-side validation

3. **Caching**:
   - Cache API responses
   - Use Redis for session storage
   - Implement CDN for frontend

## Monitoring

1. **Application Monitoring**:
   - Use services like Sentry for error tracking
   - Implement logging with proper log levels
   - Monitor API response times

2. **Infrastructure Monitoring**:
   - Set up CloudWatch (AWS)
   - Monitor CPU and memory usage
   - Set up alerts for high error rates

## Troubleshooting

### Common Issues

**Port conflicts:**
```bash
lsof -ti:5000 | xargs kill -9  # Kill process on port 5000
```

**Permission denied:**
```bash
chmod +x run_app.sh
```

**Module not found:**
```bash
pip install -r requirements.txt --force-reinstall
```

## Backup and Restore

### Backup Model Weights

```bash
# Local
cp -r backend/saved_models/ backup/

# AWS S3
aws s3 sync backend/saved_models/ s3://your-bucket/models/
```

### Restore

```bash
# Local
cp -r backup/ backend/saved_models/

# AWS S3
aws s3 sync s3://your-bucket/models/ backend/saved_models/
```

## Scaling

For high-traffic applications:

1. **Use load balancer** (AWS ELB, Nginx)
2. **Implement queue system** (Celery + Redis)
3. **Containerize with Kubernetes**
4. **Use managed services** (AWS Lambda, Google Cloud Run)

## Support

For deployment issues:
- Check application logs
- Review server error logs
- Consult cloud provider documentation
- Open an issue on GitHub

---

**Ready to deploy!** Choose the platform that best fits your needs and follow the steps above.
