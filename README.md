# OutbreakVision
Hacklytics 2025

**To view this project, run app.py**

To run setup.sh, enter this in the command prompt:


chmod +x setup.sh  # Make it executable (only once)
./setup.sh



View requirements.txt to find list of python installations necessary to run this project.


The idea for OutbreakVision came from witnessing how pandemics like COVID-19 affected different countries in vastly different ways. While some nations recovered quickly, others struggled for years. We set out to build a tool that could predict the impact of future outbreaks based on key socioeconomic factors. 
Using data from the World Bank and other sources, we incorporated factors like healthcare access, GDP per capita, pollution levels, and literacy rates to develop an Impact Score. We then trained a Random Forest Regression model to estimate the severity of an outbreak’s effects on each country. A key challenge was dealing with inconsistent and incomplete data—country names didn’t always match across datasets, requiring extensive cleaning, merging, and normalization to create a usable dataset.

Beyond predictive modeling, we wanted to visualize the impact over time. To achieve this, we built a geospatial simulation using GeoPandas, OpenCV, and Matplotlib, which generates an animated map showing how an outbreak’s impact evolves. Initially, we planned for users to download these videos, but we later integrated them directly into the web app for real-time viewing.
Another critical aspect of our project is early outbreak detection using Google Trends. Traditional disease surveillance relies on hospital reports and lab confirmations, often delaying response efforts. However, people search for symptoms online before seeking medical help, making search trends a powerful early indicator of potential outbreaks. Our system analyzes searches related to symptoms and treatments to detect surges before official case numbers rise.

One of the biggest hurdles was processing Google Trends data to extract meaningful patterns. Training a reliable machine learning model was also difficult, as search trends can be noisy and influenced by external factors like media coverage. Additionally, since we aimed to predict future outbreaks, we lacked real-world data to validate our predictions, making fine-tuning model accuracy a challenge.
Despite these obstacles, we developed a tool that can predict, measure, and visualize the impact of a pandemic on any country using real-world data. Moving forward, we aim to refine our model by integrating real-time data sources, such as Google Trends, to improve early detection and enhance predictions with government response data. OutbreakVision has the potential to help policymakers and researchers better prepare for and respond to future crises before they happen.
