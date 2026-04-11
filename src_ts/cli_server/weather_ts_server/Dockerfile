FROM node:20-alpine

WORKDIR /app

# Install only weather server dependencies
COPY src_ts/cli_server/weather_ts_server/package.json ./
RUN npm install

# Copy only weather server source files
COPY src_ts/cli_server/weather_ts_server/tsconfig.json ./
COPY src_ts/cli_server/weather_ts_server/weather.ts ./
COPY src_ts/cli_server/weather_ts_server/weather_helpers.ts ./
COPY src_ts/cli_server/weather_ts_server/weather_cli.ts ./

RUN npm run build

EXPOSE 8000
CMD ["npm", "run", "start"]

