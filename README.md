# NutriSoil - Soil Analysis and Crop Recommendation System

## Overview
NutriSoil is a web application that helps farmers and gardeners analyze their soil properties, get soil type and growth condition predictions, and receive fertilizer and crop recommendations.

## Features
- User authentication system
- Soil analysis based on key parameters (pH, NPK, moisture, EC, temperature)
- Machine learning models for soil type and growth condition prediction
- Fertilizer and crop recommendations
- AI chatbot assistant for agricultural advice

## Installation

### Prerequisites
- Python 3.8+
- Firebase account (for authentication and database)
- Google AI API key (for Gemini model)

### Setup
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with the required environment variables (see below)
4. Run the training script: `python train_models.py`
5. Start the application: `python app.py`

### Environment Variables
Create a `.env` file in the root directory with the following: