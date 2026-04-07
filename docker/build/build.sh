docker build -t preetp724/app_backend:latest -f backend.dockerfile ../..
docker build -t preetp724/app_frontend:latest -f frontend.dockerfile ../..
docker push preetp724/app_backend:latest
docker push preetp724/app_frontend:latest