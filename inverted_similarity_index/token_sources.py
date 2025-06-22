import requests
import io
from typing import Iterator, Optional


class GutenbergTextReader:
    """
    A class to fetch and read Project Gutenberg texts by paragaph.
    Intended for generating experemental data for text embeddings.
    """

    __slots__ = ("url", "_text_content")

    def __init__(self, url: str):
        self.url = url
        self._text_content: Optional[str] = None
        self._fetch_text()

    def _fetch_text(self) -> None:
        """
        Fetch the text content in one shot. Text files are pretty small.
        """
        try:
            response = requests.get(self.url)
            response.raise_for_status()

            # Project Gutenberg text files are typically UTF-8 encoded
            response.encoding = response.apparent_encoding or "utf-8"
            self._text_content = response.text

            print("Successfully fetched text from {}".format(self.url))
            print("Text length: {} characters".format(len(self._text_content)))

        except requests.RequestException as e:
            print(f"Error fetching text: {e}")
            raise

    def read_lines(self) -> Iterator[str]:
        """
        Generator that yields lines from the fetched text.
        This is where removing book-specific filler lines could be implemented
        """
        if self._text_content is None:
            raise ValueError("Text not fetched yet. Call fetch_text() first.")

        # Use StringIO to treat the string as a file-like object
        text_io = io.StringIO(self._text_content)

        for line in text_io:
            val = line.rstrip("\n\r")
            val_stripped = val.replace(" ", "")
            if val_stripped == "":
                continue
            yield val

    def read_chunks(self) -> Iterator[str]:
        """
        Return paragraphs-level chunks as strings. This is what the embeddings
        will be built on.
        """
        # Later it'd be interesting to include some peripheral
        # text and see if that changes much. Likewise with different types of
        # writing.

        data = io.StringIO(self._text_content)

        buf = []
        for line in data:
            val = line.strip()
            if len(val) == 0:
                if len(buf) > 0:
                    joined = "".join(buf)
                    # clear buffer
                    buf = []
                    yield joined
                else:
                    continue
            else:
                buf.append(line)

    def get_stats(self) -> dict:
        """
        @brief Get basic statistics about the text.
        """
        if self._text_content is None:
            return {"error": "Text not fetched yet"}

        lines = list(self.read_lines())
        return {
            "total_characters": len(self._text_content),
            "total_lines": len(lines),
            "non_empty_lines": len([line for line in lines if line.strip()]),
            "average_line_length": (
                sum(len(line) for line in lines) / len(lines) if lines else 0
            ),
        }


def get_chosen_texts() -> list[str]:
    """
    @brief Return a list of URLs for texts to be used in experiments.
    """
    # Book sources:
    # A range of topics, and extra depth/overlap in specific areas to
    # see what clusting looks like with real data. For example I expect
    # embeddings from the transcendentalist texts to nearby in latent
    #
    # I expect something like math to produce a distinct set of clusters:
    # the contents, voice(s) used in delivery, and delivery format are
    # distinct.
    #
    # Expected Clusters:
    # - Transcendentalism:
    #   - Emerson, Thoreau, Hawthorne etc
    #
    # - Children's literature, surreal fiction
    #   - Alice's Adventures in Wonderland etc
    #
    # Other intended partial semantic overlap:
    # - forestry info <- nature, sustainability-> transcendentalism
    # - Shackleton's expedition <- individualism, expidition -> transcendentalism
    # - The scarlet letter <- boston, timeframe -> Walden

    return [
        "https://www.gutenberg.org/cache/epub/11/pg11.txt",  # Alice's Adventures in Wonderland
        "https://www.gutenberg.org/cache/epub/5200/pg5200.txt",  # The Metamorphosis, Kafka
        "https://www.gutenberg.org/cache/epub/7849/pg7849.txt",  # The Trial, Kafka
        "https://www.gutenberg.org/cache/epub/25344/pg25344.txt",  # The Scarlet Letter
        "https://www.gutenberg.org/cache/epub/205/pg205.txt",  # Walden: Life in the Woods
        "https://www.gutenberg.org/cache/epub/16643/pg16643.txt",  # Essays of Ralph Waldo Emerson
        "https://www.gutenberg.org/cache/epub/29433/pg29433.txt",  # Nature by Ralph Waldo Emerson
        "https://www.gutenberg.org/cache/epub/1232/pg1232.txt",  # The Prince
        "https://www.gutenberg.org/cache/epub/66944/pg66944.txt",  # The Principle of Relativity
        "https://www.gutenberg.org/cache/epub/48874/pg48874.txt",  # A brief history of forestry
        "https://www.gutenberg.org/cache/epub/5199/pg5199.txt",  # South, the Story of Shackleton's Last Expedition
    ]


def main():
    """
    Example usage of the GutenbergTextReader class.
    """

    urls = get_chosen_texts

    # Create reader instance
    chunk_count = 0

    for url in urls:
        reader = GutenbergTextReader(url)

        # Display basic statistics
        stats = reader.get_stats()
        print("\nText Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        for chunk in reader.read_chunks():
            print("Chunk {}:{}".format(chunk_count, chunk))

            chunk_count += 1
            if chunk_count > 30:
                break

    chunk_count = 0
    for chunk in reader.read_chunks():
        print("Chunk {}:{}".format(chunk_count, chunk))

        chunk_count += 1
        if chunk_count > 30:
            break


if __name__ == "__main__":
    main()
