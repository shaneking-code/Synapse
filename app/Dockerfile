# Stage 1: Build the React app using Vite
FROM node:16-alpine AS builder

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

# Stage 2: Serve the built static assets with Nginx
FROM nginx:alpine

# Copy the production build into the Nginx html folder
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]