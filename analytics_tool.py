import streamlit as st
import pandas as pd
import spacy
from textblob import TextBlob
import plotly.express as px
from collections import Counter
import re
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer, AutoModel

# Load data
def load_data(file_path, text_column="Text"):
  data = pd.read_excel(file_path)
  if text_column not in data.columns:
      raise ValueError(f"Dataset must contain a '{text_column}' column.")
  data.dropna(subset=[text_column], inplace=True)
  return data

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

### RIKA START ###
# Function to classify sentiment as 'positive', 'neutral', or 'negative'
def classify_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Function to extract companies and their sentiment
def extract_companies_with_sentiment(text_column, nlp_model, exclude_list=None):
    print(f"nlp_model type: {type(nlp_model)}")
    exclude_list = set(exclude_list or [])  # Ensure exclude_list is a set for faster lookup
    org_sentiment = []

    for text in text_column:
        doc = nlp_model(text)
        sentiment = classify_sentiment(text)

        # Extract unique organizations in the row
        unique_orgs = {
            ent.text.strip()
            for ent in doc.ents
            if ent.label_ == "ORG" and ent.text.strip() not in exclude_list
        }

        # Append each unique organization with its sentiment
        org_sentiment.extend([(org, sentiment) for org in unique_orgs])

    return org_sentiment

# Function to plot the 100% stacked bar chart
def plot_top_10_orgs_with_sentiment(org_sentiment):
    # Convert to DataFrame
    df = pd.DataFrame(org_sentiment, columns=["Organization", "Sentiment"])

    # Count occurrences by organization and sentiment
    org_counts = df.groupby(["Organization", "Sentiment"]).size().reset_index(name="Count")

    # Find the top 10 most common organizations
    top_10_orgs = df["Organization"].value_counts().head(10).index
    org_counts = org_counts[org_counts["Organization"].isin(top_10_orgs)]

    # Calculate the total counts per organization
    org_counts["Total"] = org_counts.groupby("Organization")["Count"].transform("sum")

    # Calculate percentages for 100% stacked bar chart
    org_counts["Percentage"] = (org_counts["Count"] / org_counts["Total"]) * 100 

    # Sort organizations alphabetically
    org_counts = org_counts.sort_values(by="Organization").drop(columns=["Total"])

    # Plot interactive bar chart using Plotly
    fig = px.bar(
        org_counts,
        x="Organization",
        y="Percentage",
        color="Sentiment",
        color_discrete_map={"Positive": "lightgreen", "Neutral": "skyblue", "Negative": "salmon"},
        hover_data={"Organization": True, "Sentiment": True, "Percentage": True},
        title="Sentiment of Top 10 Organizations",
        labels={"Organization": "Organization", "Percentage": "Percentage of Sentiment"},
    )

    # Show the interactive chart with corrected hover formatting
    fig.update_traces(
        marker=dict(line=dict(width=1, color="black")),
        hovertemplate="Organization: %{x}<br>Sentiment: %{customdata[0]}<br>Percentage: %{y:.0f}%",
    )
    fig.update_layout(
        xaxis_title="Organization",
        yaxis_title="Percentage",
        showlegend=True,
    )

    # Return the figure to be used in Streamlit
    return fig


# Function to get polarity score using TextBlob
def get_polarity(text):
    # Classifying polarity: TextBlob returns polarity between -1 and 1
    return TextBlob(text).sentiment.polarity

# Function to extract countries from the text using spaCy's NER
def extract_countries(text):
    doc = nlp(text)
    countries = set()
    for ent in doc.ents:
        if ent.label_ == "GPE":  # GPE stands for Geo-Political Entity (countries, cities, etc.)
            countries.add(ent.text)
    return countries

# Function to map countries to continents
def map_countries_to_continents(data, continent_dict):
    # Add a column for continent category
    data['Continent'] = data['Countries'].map(continent_dict)
    # Remove rows where the continent is NaN (i.e., countries not in the dictionary)
    return data.dropna(subset=['Continent'])

# Function to calculate mean polarity score per continent
def calculate_mean_polarity_by_continent(data):
    # Group by continent and calculate the mean polarity score
    return data.groupby('Continent')['Polarity'].mean().reset_index()

# Function to plot the bar chart (sorted by score)
def plot_polarity_by_continent(continent_polarity):
    # Sort by polarity score in descending order
    continent_polarity = continent_polarity.sort_values(by="Polarity", ascending=False)
    
    # Plot
    fig = px.bar(
        continent_polarity,
        x="Continent",
        y="Polarity",
        color="Continent",
        title="Mean Polarity Score by Continent",
        labels={"Polarity": "Mean Polarity Score", "Continent": "Continent"},
        color_discrete_sequence=px.colors.qualitative.Set2,
        text_auto=".4f"
    )
    # fig.show()
    return fig

### RIKA END ###

### PRITIKA START ###
def extract_country(text):
    """
    Extracts the country mentioned in the text using SpaCy's named entity recognition (NER).
    """
    doc = nlp(text)
    countries = []
    for ent in doc.ents:
        if ent.label_ == "GPE":  # GPE stands for Geopolitical Entity (countries, cities, etc.)
            country = ent.text
            # Normalize "the United States", "United States", "US", and "U.S." as "United States"
            if country.lower() in ["united states", "the united states", "us", "u.s."]:
                country = "United States"
            countries.append(country)
    return countries

def get_sentiment(text):
    analysis = TextBlob(text)
    return "Positive" if analysis.sentiment.polarity > 0 else "Neutral" if analysis.sentiment.polarity == 0 else "Negative"

def extract_crime_month(text):
    """
    Extracts the month of the crime from the text using regular expressions.
    Handles both full month names and short forms.
    """
    month_pattern = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    match = re.search(month_pattern, text, re.IGNORECASE)

    if match:
        month_str = match.group(1).capitalize()
        month_mapping = {
            "Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April",
            "May": "May", "Jun": "June", "Jul": "July", "Aug": "August",
            "Sep": "September", "Oct": "October", "Nov": "November", "Dec": "December"
        }
        return month_mapping.get(month_str, month_str)
    return None  # Return None if no month is found

def process_excel_file_1(file_path):
    """
    Processes the Excel file to extract country occurrences and sentiment.
    Creates a stacked bar graph showing the sentiment distribution for the top 10 countries, ordered in descending order.
    """
    try:
        # Read the Excel file into a DataFrame
        df = pd.read_excel(file_path)

        # Ensure the 'Text' column exists
        if "Text" not in df.columns:
            raise ValueError("The Excel file must contain a 'Text' column.")

        # Extract country and sentiment data
        country_data = []
        for text in df["Text"].dropna():
            countries = extract_country(text)
            sentiment = get_sentiment(text)
            for country in countries:
                country_data.append([country, sentiment])

        if not country_data:
            print("No valid data found.")
            return

        # Create DataFrame for analysis
        country_df = pd.DataFrame(country_data, columns=["Country", "Sentiment"])

        # Count occurrences of each country
        country_counts = country_df["Country"].value_counts()

        # Select top 10 countries
        top_10_countries = country_counts.head(10).index
        top_10_df = country_df[country_df["Country"].isin(top_10_countries)]

        # Pivot table for stacked bar chart
        sentiment_counts = top_10_df.pivot_table(index="Country", columns="Sentiment", aggfunc="size", fill_value=0)

        # Order by total occurrences in descending order
        sentiment_counts = sentiment_counts.loc[sentiment_counts.sum(axis=1).sort_values(ascending=False).index]

        # Create a plotly stacked bar chart
        fig = px.bar(sentiment_counts, 
                     x=sentiment_counts.index, 
                     y=sentiment_counts.columns, 
                     title="Sentiment Distribution in Top 10 Countries",
                     labels={"value": "Occurrences", "Sentiment": "Sentiment", "Country": "Country"},
                     color_discrete_map={
                         'negative': 'lightcoral',
                         'neutral': 'lightblue',
                         'positive': 'lightgreen'
                     })

        # Show the figure
        fig.update_layout(barmode='stack')
        # fig.show()
        return fig

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def process_excel_file_2(file_path):
    """
    Processes the Excel file to extract crime occurrences by month and sentiment.
    Creates a stacked bar graph showing sentiment distribution per month.
    """
    try:
        df = pd.read_excel(file_path)

        if "Text" not in df.columns:
            raise ValueError("The Excel file must contain a 'Text' column.")

        # Extract month and sentiment
        month_sentiment_data = []
        for text in df["Text"].dropna():
            month = extract_crime_month(text)
            if month:  # Exclude "No Month"
                sentiment = get_sentiment(text)
                month_sentiment_data.append([month, sentiment])

        if not month_sentiment_data:
            print("No valid data found.")
            return

        # Create DataFrame for analysis
        month_sentiment_df = pd.DataFrame(month_sentiment_data, columns=["Month", "Sentiment"])

        # Ensure months appear in chronological order
        month_order = ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]
        month_sentiment_df["Month"] = pd.Categorical(month_sentiment_df["Month"], categories=month_order, ordered=True)

        # Count occurrences of each sentiment per month
        sentiment_counts = month_sentiment_df.groupby(["Month", "Sentiment"]).size().unstack(fill_value=0)

        # Create a Plotly stacked bar chart
        fig = px.bar(sentiment_counts, 
                     x=sentiment_counts.index, 
                     y=sentiment_counts.columns, 
                     title="Sentiment Distribution of News Articles by Month",
                     labels={"value": "Number of Articles", "Sentiment": "Sentiment", "Month": "Month"},
                     color_discrete_map={
                         'negative': 'lightcoral',
                         'neutral': 'lightblue',
                         'positive': 'lightgreen'
                     })

        # Update layout for better presentation
        fig.update_layout(barmode='stack',
                          xaxis_title="Month",
                          yaxis_title="Number of Articles",
                          xaxis_tickangle=45)

        return fig

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

### PRITIKA END ###

### HUAN YAO START ###

def resolve_pronouns(doc):
  try:
      resolved_text = doc._.coref_resolved if hasattr(doc._, 'coref_resolved') else doc.text
  except Exception:
      resolved_text = []
      previous_entities = []
      for token in doc:
          if token.pos_ == "PRON":
              resolved_text.append(previous_entities[-1] if previous_entities else token.text)
          else:
              resolved_text.append(token.text)
              if token.ent_type_ or token.pos_ == "NOUN":
                  previous_entities.append(token.text)
      resolved_text = " ".join(resolved_text)
  return resolved_text


def extract_svo_triples(text, nlp_model):
   doc = nlp_model(text)
   triples = set()
   for sent in doc.sents:
       subject, verb, obj = None, None, None
       for token in sent:
           if 'subj' in token.dep_ and token.ent_type_ in {"PERSON", "ORG", "GPE", "LOC"}:
               subject = token.text
           if token.pos_ == 'VERB' and token.text.lower() not in ["naysaid", "used"]:  # Filter out unwanted verbs
               verb = token.text
           if 'obj' in token.dep_:
               obj = token.text
               # Stop processing after object is found to extract from left to right
               if subject and verb and obj:
                   triples.add((subject, verb, obj))
                   continue # Exit the loop to avoid relation after object
   return list(triples)


def encode_text(text, model, tokenizer):
  """Encodes text into an embedding using the specified model and tokenizer."""
  inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
  with torch.no_grad():
      outputs = model(**inputs)
  return outputs.last_hidden_state.mean(dim=1).squeeze()


def deduplicate_relations(triples, text, embedding_model, tokenizer, similarity_threshold=0.85):
   """Deduplicate triples by retaining the one most aligned with the context of the text."""
   context_embedding = encode_text(text, embedding_model, tokenizer)


   # Set to track already added triples
   seen_triples = set()
   deduplicated_triples = []


   # Group by (subject, object) to identify duplicates based on subject and object
   grouped_triples = {}
   for triple in triples:
       key = (triple[0], triple[2])  # (subject, object)
       if key not in grouped_triples:
           grouped_triples[key] = []
       grouped_triples[key].append(triple)


   # Process each group and add the best triple to the final list
   for key, triple_group in grouped_triples.items():
       if len(triple_group) == 1:
           # Only one triple exists for this key
           best_triple = triple_group[0]
       else:
           # Multiple triples, compute similarity and pick the best one
           best_triple = None
           best_similarity = -1
           for triple in triple_group:
               triple_text = f"{triple[0]} {triple[1]} {triple[2]}"
               triple_embedding = encode_text(triple_text, embedding_model, tokenizer)
               similarity = cosine_similarity(
                   context_embedding.unsqueeze(0), triple_embedding.unsqueeze(0)
               ).item()
               if similarity > best_similarity:
                   best_triple = triple
                   best_similarity = similarity


       # Only add to deduplicated list if not already seen
       if best_triple not in seen_triples:
           deduplicated_triples.append(best_triple)
           seen_triples.add(best_triple)


   # Group by (subject, relation) to further deduplicate based on subject and relation
   grouped_triples = {}
   for triple in deduplicated_triples:
       key = (triple[0], triple[1])  # (subject, relation)
       if key not in grouped_triples:
           grouped_triples[key] = []
       grouped_triples[key].append(triple)


   # Process each group and add the best triple to the final list (again)
   deduplicated_triples = []  # Reset deduplicated_triples for this stage
   seen_triples = set()  # Reset seen_triples for this stage
   for key, triple_group in grouped_triples.items():
       if len(triple_group) == 1:
           best_triple = triple_group[0]
       else:
           best_triple = None
           best_similarity = -1
           for triple in triple_group:
               triple_text = f"{triple[0]} {triple[1]} {triple[2]}"
               triple_embedding = encode_text(triple_text, embedding_model, tokenizer)
               similarity = cosine_similarity(
                   context_embedding.unsqueeze(0), triple_embedding.unsqueeze(0)
               ).item()
               if similarity > best_similarity:
                   best_triple = triple
                   best_similarity = similarity


       if best_triple not in seen_triples:
           deduplicated_triples.append(best_triple)
           seen_triples.add(best_triple)


   return deduplicated_triples


def is_valid_relation(subject, verb, obj, context_keywords):
  synonyms = set()
  for kw in context_keywords:
      synonyms.update(syn.name().split('.')[0] for syn in wn.synsets(kw))
  relevant = any(kw in subject.lower() or kw in obj.lower() for kw in synonyms)
  return relevant


def align_relationships(triples, context_keywords, text, embedding_model, tokenizer, nlp_model, similarity_threshold=0.9):
  """Filter and deduplicate triples based on relevance and semantic similarity."""
  valid_triples = []
  for triple in triples:
      if is_valid_relation(*triple, context_keywords):
          valid_triples.append(triple)
  return deduplicate_relations(valid_triples, text, embedding_model, tokenizer, similarity_threshold)


def summarize_text(text, nlp_model):
  doc = nlp_model(text)
  sentences = list(doc.sents)
  return sentences[0].text if sentences else ""


def store_relationships(triples, row_id):
  return pd.DataFrame(triples, columns=["Subject", "Relation", "Object"]).assign(Row=row_id)


def process_text(text, nlp_model, embedding_model, tokenizer):
  """Process a text to extract and align relationships."""
  doc = nlp_model(text)
  summary = summarize_text(text, nlp_model)
  context_keywords = [token.text.lower() for token in nlp_model(summary) if token.is_alpha]
  resolved_text = resolve_pronouns(doc)
  triples = extract_svo_triples(resolved_text, nlp_model)
  return align_relationships(triples, context_keywords, text, embedding_model, tokenizer, nlp_model)


def display_er_diagram(data, nlp_model, embedding_model, tokenizer, max_rows=None):
  """Display extracted relationships and print deduplicated results."""
  all_relationships = pd.DataFrame(columns=["Subject", "Relation", "Object", "Row"])
  for idx, text in enumerate(data['Text']):
      if max_rows and idx >= max_rows:
          break
      print(f"\nRow {idx}:\nText: {text}")
      relationships = process_text(text, nlp_model, embedding_model, tokenizer)
      relationships_df = store_relationships(relationships, idx)
      all_relationships = pd.concat([all_relationships, relationships_df], ignore_index=True)
      print("Extracted Relationships (Deduplicated):")
      for rel in relationships:
          print(f"  {rel[0]} --({rel[1]})--> {rel[2]}")
  return all_relationships

def main1(file_path, text_column="Text", max_rows=None):
#   nlp = spacy.load("en_core_web_sm")
  tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
  embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
  data = load_data(file_path, text_column)
  relationships = display_er_diagram(data, nlp, embedding_model, tokenizer, max_rows=max_rows)
  relationships.to_csv("extracted_relationships.csv", index=False)

### HUAN YAO END ###


### YEE TENG START ###
def extract_entities(text, nlp_model):
    """Extract and categorize entities from the text."""
    doc = nlp_model(text)
    entities = []
   
    for ent in doc.ents:
        # Filter for specific entity categories
        if ent.label_ in ["ORG", "PERSON", "GPE", "PRODUCT"]:
            entities.append((ent.text, ent.label_))
   
    return entities


def process_data(file_path, text_column="Text", max_rows=10):
    """Process the first 'max_rows' rows of data, extract and categorize entities."""
    nlp = spacy.load("en_core_web_sm")  # Load pre-trained spaCy model for NER
    data = load_data(file_path, text_column)
   
    # Limit the rows to process
    data = data.head(max_rows)
   
    results = []
    for idx, row in data.iterrows():
        text = row[text_column]
        entities = extract_entities(text, nlp)
       
        # Add the row number, entity, and category to the results list
        for entity, category in entities:
            results.append({"Row": idx + 1, "Entity": entity, "Category": category})  # Using idx+1 for 1-based row indexing
   
    # Convert the results into a DataFrame
    results_df = pd.DataFrame(results)
   
    # Remove duplicates based on the 'Entity' column
    results_df = results_df.drop_duplicates(subset=["Entity"])
   
    return results_df


def plot_entity_distribution(results_df):
    """Plot an interactive pie chart for the distribution of entity categories using Plotly."""
    category_counts = results_df["Category"].value_counts().reset_index()
    category_counts.columns = ["Category", "Count"]  # Renaming columns for clarity

    # Define colors (Plotly uses automatic color mapping if not specified)
    colors = ['#FF85B3', '#1E90FF', '#FFD700', '#FFA500', '#8A2BE2', '#00CED1', '#FF6347', '#32CD32']

    # Create interactive Plotly pie chart
    fig = px.pie(
        category_counts,
        names="Category",
        values="Count",
        title="Entity Category Distribution (ORG, PERSON, GPE, PRODUCT)",
        color="Category",
        color_discrete_sequence=colors[:len(category_counts)],  # Limit colors to the number of categories
        hole=0.3  # Donut-style chart (set to 0 for full pie)
    )

    # Improve layout and interactivity
    fig.update_traces(textinfo="percent+label", hoverinfo="label+value+percent")
    fig.update_layout(title_font=dict(size=20, family="Arial", color="black"))

    # Display the chart
    # fig.show()
    return fig

def main2(file_path, text_column="Text"):
    """Main function to run the extraction and categorization for the first 10 rows and plot the pie chart."""
    results_df = process_data(file_path, text_column, max_rows=10)
    results_df.to_csv("categorized_entities_filtered.csv", index=False)
    print("Entity categorization complete for the first 10 rows! Results saved to 'categorized_entities_filtered.csv'.")
   
    # Plot pie chart for entity categories
    return plot_entity_distribution(results_df)

### YEE TENG END ###

# Streamlit app
st.title("Visualisation Tool")
st.write("Upload a dataset to generate visualisation charts.")

# File upload
uploaded_file = st.file_uploader("Upload an Excel file with a 'Text' column", type=["xlsx"])

if uploaded_file:
    # Load the data
    data = pd.read_excel(uploaded_file)

    if "Text" not in data.columns:
        st.error("The file must contain a 'Text' column.")
    else:
        ### RIKA START ###
        # Exclusion list for unwanted terms
        exclude_list = ["Reuters", "CNA", "CNN", "The Straits Times", "COVID-19", "AI"]
        # Extract organizations with sentiment
        org_sentiment = extract_companies_with_sentiment(data["Text"], nlp, exclude_list)
        # Display the sentiment analysis results
        if org_sentiment:
            # Plot the interactive bar chart of top organizations by sentiment
            fig = plot_top_10_orgs_with_sentiment(org_sentiment)
            st.plotly_chart(fig)  # Display the chart

    
        # Load the dataset
        data = load_data(uploaded_file)
        # Apply polarity function to the 'Text' column
        data['Polarity'] = data['Text'].apply(get_polarity)
        # Extract countries from the text
        data['Countries'] = data['Text'].apply(extract_countries)
        # Flatten the DataFrame to one row per country mentioned in the text
        expanded_data = data.explode('Countries')
        # Mapping countries to continents
        continent_dict = {
        "USA": "North America", "United States": "North America", "United States of America": "North America", "US": "North America",
        "Canada": "North America", "Mexico": "North America",
        "Brazil": "South America", "Argentina": "South America", "Colombia": "South America",
        "India": "Asia", "China": "Asia", "Japan": "Asia", "South Korea": "Asia",
        "Germany": "Europe", "France": "Europe", "Italy": "Europe",
        "Australia": "Oceania", "New Zealand": "Oceania"
        }
        # Map countries to continents and clean the data
        expanded_data = map_countries_to_continents(expanded_data, continent_dict)
        # Calculate mean polarity score per continent
        continent_polarity = calculate_mean_polarity_by_continent(expanded_data)
        # Plot the mean polarity by continent
        continent_fig = plot_polarity_by_continent(continent_polarity)
        st.plotly_chart(continent_fig)
        ### RIKA END ###


        ### PRITIKA START ###
        # Main function
        if __name__ == "__main__":
            fig1 = process_excel_file_1(uploaded_file)
            st.plotly_chart(fig1)

            fig2 = process_excel_file_2(uploaded_file)
            st.plotly_chart(fig2)
        ### PRITIKA END ###


        ### HUAN YAO START ###
        main1(uploaded_file, max_rows=10)
        ### HUAN YAO END ###


        ### YEE TENG START ###
        figure = main2(uploaded_file)
        st.plotly_chart(figure)
        ### YEE TENG END ###

