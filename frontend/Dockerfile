# Syntra AI — dashboard frontend, built as a static bundle served by nginx.

FROM node:20-slim AS build
WORKDIR /srv/app
COPY package.json .
RUN npm install
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /srv/app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
