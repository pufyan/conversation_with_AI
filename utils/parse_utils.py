import requests

from bs4 import BeautifulSoup

from embedchain.utils import clean_string

import re


from langchain.document_loaders import YoutubeLoader


class YoutubeVideoLoader:

    def load_data(self, url):
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True, language='ru')
        doc = loader.load()
        output = []
        if not len(doc):
            raise ValueError("No data found")
        content = doc[0].page_content
        content = clean_string(content)
        meta_data = doc[0].metadata
        meta_data["url"] = url
        output.append({
            "content": content,
            "meta_data": meta_data,
        })
        return output


class WebPageLoader:

    def load_data(self, url):

        # is youtube video:
        if url.startswith('https://www.youtube.com') or url.startswith('https://youtu.be') or url.startswith('https://m.youtube.com'):
            return YoutubeVideoLoader().load_data(url)
        try:
            response = requests.get(url)
        except requests.exceptions.MissingSchema:
            print(f'MissingSchema: {url}')
            return
        except requests.exceptions.InvalidURL:
            print(f'Invalid url: {url}')  # TODO add web search
            return

        data = response.content
        soup = BeautifulSoup(data, 'html.parser')
        for tag in soup([
            "nav", "aside", "form", "header",
            "noscript", "svg", "canvas",
            "footer", "script", "style"
        ]):
            tag.string = " "
        output = []
        content = soup.get_text()
        content = clean_string(content)
        return content


def is_link_only(content):
    if re.match(re.compile(r'^((?:http|ftp)s?://)|([\w]{2,}\.[\w]{2,})$', re.IGNORECASE), content):
        # link to parse
        valid_link = 'https://' + content if not content.startswith('http') else content
        return valid_link


def get_links(content):

    # find all links in the text:
    links = LINK_RE.findall(content)
    links = [link[0] + link[1] for link in links]
    return links
