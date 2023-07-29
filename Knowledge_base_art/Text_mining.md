Text Mining: Analyzing text about machine learning.
synaptical.comAccording to Wikipedia:
"Text mining, also referred to as text data mining, similar to text analytics, is the process of deriving high-quality information from text. It involves the discovery by computer of new, previously unknown information, by automatically extracting information from different written resources."
Further reading: Text mining - Wikipedia
Technology /human development as seen a large growth over the years which has help humans and other species stayed over the edge(survive). In area of AI/machine learning, text mining/analyst has become one of the major ways to solve business problems(or problems in general) by applying statistical linguistic and machine learning techniques that model and structure textual data into business intelligence, EDA, research and development. Text mining is used in areas such as Software app, Biomedical, Sentiment analysis, Business Intelligence.
In this Example, Let's try to discover more related concepts by doing some text mining. We will start with a text about Machine Learning, extract keywords from it, and then try to visualize the result.
Getting the data.

The first step in every data science/ML project is getting the data to work with. here we will use 'request' to get the data from Wikipedia which will be inform of an HTML.
import requests 
url = 'https://en.wikipedia.org/wiki/Machine_learning'
text = requests.get(url).content.decode('utf-8')
print(text[:1000])
2. Transforming the Data.
we have imported hour data in an HTML code form, so we need to transform our data into a more structured form. In this case we are goin to use the built -in HTMLParser object from python.
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    script = False
    res = ''
    def handle_starttag(self, tag, attrs):
        if tag.lower() in ['script', 'style']:
            self.script = True
    def handle_endtag(self, tag):
        if tag.lower() in ['script', 'style']:
            self.script = False
    def handle_data(self, data):
        if str.strip(data) == '' or self.script:
            return
        self.res += ' '+data.replace('[ edit ]','')
parser = MyHTMLParser()
parser.feed(text)
text = parser.res
print(text[:1000])
3. Insights.
The most important step is to turn our data into some form from which we can draw insights. In our case, we want to extract keywords from the text, and see which keywords are more meaningful. We will use Python library called RAKE for keyword extraction.
import sys
!{sys.executable} -m pip install nlp_rake
import nlp_rake
extractor = nlp_rake.Rake(max_words=2, min_freq=2,min_chars=13,)
res = extractor.apply(text)
res
With the above method, we were able to generate a list with most common word related to machine learning. Let's further power this result by visualizing it.
4. Vizualization
WorldCloud is a great tools that can be use to visualize word frequency.
!{sys.executable} -m pip install wordcloud
from wordcloud import WordCloud
wc = WordCloud(background_color='white', width=800, height=600)
plt.figure(figsize=(15,7))
plt.imshow(wc.generate_from_frequencies({k:v for k,v in res}))
WordCloud object is responsible for taking in either original text, or pre-computed list of words with their frequencies, and returns and image, which can then be displayed using matplotlib.
Let's try plotting the original text which as no frequency:
plt.figure(figsize=(15,8))
plt.imshow(wc.generate(text))
5. Conclusion.
You can see that word cloud now looks more impressive, but it also contains a lot of noise (eg. unrelated words such as ISBN). Also, we get fewer keywords that consist of two words, e.g. computer science. This is because RAKE algorithm does much better job at selecting good keywords from text. This example illustrates the importance of data pre-processing and cleaning, because clear picture at the end will allow us to make better decisions.
Bonus.
we can save our result as png image.
wc.generate_from_frequencies({k:v for k,v in res}).to_file('ml_worldcl.png')
Futher reading:
Data Science for Beginners (microsoft.github.io)
