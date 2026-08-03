# VITAL Frontend Dockerfile
FROM node:18-alpine

WORKDIR /app

# Copy dependency manifest files
COPY package*.json ./

# Install npm dependencies
RUN npm ci

# Copy frontend source files
COPY . .

# Build Next.js production bundle
RUN npm run build

# Expose Next.js server port
EXPOSE 3000

# Start production server
CMD ["npm", "run", "start"]
