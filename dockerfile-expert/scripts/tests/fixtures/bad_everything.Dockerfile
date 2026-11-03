FROM ubuntu
MAINTAINER someone@example.com
ENV DEBIAN_FRONTEND=noninteractive
ENV API_TOKEN=abc123
ARG DB_PASSWORD
ARG CONFIG_PATH=/etc/app
RUN apt-get update
RUN apt-get install build-essential curl
RUN curl -fsSL https://example.com/install.sh | bash
RUN sudo chmod -R 777 /opt
WORKDIR app
ADD src/ /app/src/
COPY .env /app/.env
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
RUN apk add jq
RUN npm install
RUN make all && pip install requests
CMD ["echo", "first"]
CMD npm start
