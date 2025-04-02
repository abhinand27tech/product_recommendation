# Supermarket Recommendation System

A machine learning-based recommendation system for supermarkets that provides personalized product recommendations to customers.

## Features

- User-based collaborative filtering
- Item-based collaborative filtering
- Association rules mining
- Support for both registered and anonymous customers
- Godown-specific recommendations
- Real-time recommendation generation

## Prerequisites

- Docker
- Docker Compose
- Git

## Project Structure

```
.
├── backend/
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── app.py
│   ├── ml_services.py
│   └── requirements.txt
├── frontend/
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── package.json
│   └── src/
├── data/
│   ├── Header_comb.csv
│   ├── Detail_comb.csv
│   └── item_no_product_names.csv
└── docker-compose.yml
```

## Getting Started

1. Clone the repository:
```bash
git clone <repository-url>
cd supermarket-recommendation
```

2. Build and start the containers:
```bash
docker-compose up --build
```

3. Access the application:
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000

## Development

The application is set up with hot-reloading for both frontend and backend:

- Frontend changes will automatically reload in the browser
- Backend changes will automatically restart the Flask server

## Data Files

Place the following CSV files in the `data` directory:
- `Header_comb.csv`: Transaction header data
- `Detail_comb.csv`: Transaction detail data
- `item_no_product_names.csv`: Product name mappings

## API Endpoints

- `GET /api/godowns`: Get list of available godowns
- `GET /api/customers/<godown_code>`: Get list of customers for a specific godown
- `GET /api/recommendations/<cust_code>/<godown_code>`: Get recommendations for a customer

## Stopping the Application

To stop the application:

```bash
docker-compose down
```

## Troubleshooting

1. If the containers fail to start, check the logs:
```bash
docker-compose logs
```

2. To rebuild the containers after making changes:
```bash
docker-compose up --build
```

3. To remove all containers and volumes:
```bash
docker-compose down -v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 