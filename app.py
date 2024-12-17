import pandas as pd
import base64
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from flask import Flask, render_template, request
from scipy.stats import chi2_contingency

app = Flask(__name__)

# Load your dataset
dataset = pd.read_csv('C:/train.csv')
dataset['family_size'] = dataset['SibSp'] + dataset['Parch']
dataset = dataset.drop(columns=['SibSp', 'Parch'])

# Function to convert Matplotlib figures to base64


def fig_to_base64(fig):
    img_bytes = BytesIO()
    fig.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    return img_base64


@app.route('/')
def index():

    return render_template('index.html')


@app.route('/figure/<figure_type>')
def figure(figure_type):
    # Depending on the button clicked, render a different figure
    if figure_type == 'figure1':
        fig1, ax = plt.subplots()
        dataset[['Pclass', 'Survived']].groupby(
            'Pclass').mean().Survived.plot.bar(color='skyblue', ax=ax)
        ax.set_xlabel('Pclass')
        ax.set_ylabel('Survival Probability')
        ax.set_title("Survival Probability by Passenger Class")
        fig_base64 = fig_to_base64(fig1)

    elif figure_type == 'figure2':
        fig2, ax = plt.subplots()
        dataset['Survived'].value_counts().plot(
            ax=ax, kind='bar', color='blue')
        ax.set_xlabel('Survived or not')
        ax.set_ylabel('Passenger Count')
        ax.set_title('Survival Count by Category')
        fig_base64 = fig_to_base64(fig2)
    elif figure_type == 'figure3':
        vc = dataset[['Sex', 'Survived']].groupby('Sex').mean().Survived
        fig3, ax = plt.subplots()
        vc.plot(kind='bar', ax=ax, color=['#FF69B4', '#1E90FF'])
        ax.set_xlabel('Sex')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Survival Probability by Sex')
        fig_base64 = fig_to_base64(fig3)
    elif figure_type == 'figure4':
        sex_survived_counts = dataset.groupby(
            ['Sex', 'Survived']).size().unstack()

        colors = ['pink', 'royalblue']
        fig4, ax = plt.subplots(figsize=(10, 6))
        sex_survived_counts.plot.pie(
            subplots=True, autopct='%1.1f%%', startangle=90, colors=colors, legend=True, ax=ax)
        plt.suptitle("Distribution of Passengers by Sex and Survival Status")
        fig_base64 = fig_to_base64(fig4)
    elif figure_type == 'figure5':
        g = sns.FacetGrid(dataset, col='Survived', height=5, aspect=1)
        g.map(sns.histplot, 'Age', kde=True, bins=30, color='blue')
        g.set_axis_labels("Age", "Count")
        for ax in g.axes.flat:
            ax.set_xticks(range(0, 101, 10))
        g.fig.suptitle("Distribution of Survived by Age")
        g.fig.subplots_adjust(top=0.85)
        fig_base64 = fig_to_base64(g.fig)
    elif figure_type == 'figure6':
        sns.countplot(x='family_size', data=dataset, palette='Set2')
        plt.title('Distribution of Family Size')
        plt.xlabel('Family Size')
        plt.ylabel('Number of Passengers')
        fig_base64 = fig_to_base64(plt.gcf())
    # Create a contingency table between Embarked and Pclass
    elif figure_type == 'figure7':
        contingency_table = pd.crosstab(dataset['Embarked'], dataset['Pclass'])
        print("Table de contingence :")
        print(contingency_table)

        # Perform the chi-square test
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-square test results: chi2={chi2}, p={p}, dof={dof}")

        # Create the heatmap
        fig, ax = plt.subplots()
        sns.heatmap(contingency_table, annot=True, cmap='YlGnBu', ax=ax)
        ax.set_title(
            "Relationship between Embarkation Port and Ticket Class")
        fig_base64 = fig_to_base64(fig)
    # Add more elif blocks for other figures

    return render_template('figure.html', fig_base64=fig_base64)


if __name__ == '__main__':
    app.run(debug=True)
