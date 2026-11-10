# Controlled Vocabularies for Museum Object Classification

This project integrates four major controlled vocabularies to enable robust, explainable classification of museum objects using Retrieval-Augmented Generation (RAG). Each vocabulary covers a distinct domain, providing terms, definitions, and URIs for semantic interoperability.

## Broad Classification Categories

The cataloging system first classifies each object into one of seven broad, mutually exclusive categories. This initial human-led step ensures high-level accuracy and constrains the subsequent AI-assisted search to the correct vocabulary.

| Category | Description | Examples |
| :--- | :--- | :--- |
| **Musical Instrument** | Objects created or adapted to make musical sounds, including their essential functional components. | `Electric guitar`, `trumpet`, `bass drum`, `banjo`, `synthesizer` |
| **Clothing** | Items worn on the body, including garments, footwear, headwear, and costume accessories for daily wear or stage performance. | `Stage suit`, `bowler hat`, `silk dress`, `leather waistcoat` |
| **Audiovisual Media** | Physical carriers of recorded sound and/or moving images, distinguishing the object from its intellectual content. | `Vinyl LP`, `7" single`, `audio cassette`, `CD`, `master tape` |
| **Textual & Graphic Works** | Two-dimensional objects like paper whose primary function is to convey information via text, notation, or imagery. | `Sheet music`, `concert poster`, `fanzine`, `handwritten lyrics` |
| **Merchandise** | Commercially produced items created for sale to fans, often featuring an artist's name, image, or logo. | `Beatles lunchbox`, `fan club badge`, `tour programme` |
| **Personal Effects** | Objects personally owned, used, or modified by an individual that are not otherwise classified. | `John Lennon's glasses`, `guitar case`, `plectrum`, `wallet` |
| **Award & Recognition** | Official items conferred to recognize achievement or commemorate participation. | `Gold disc award`, `BRIT Award trophy`, `sales certificate` |

---
## Vocabulary Sources

### 1. Getty AAT (Art & Architecture Thesaurus)
- **Domain:** Textual & Graphic Works, Merchandise, Personal Effects, Awards & Recognition
- **Source:** Getty Vocabulary Program
- **Format:** SPARQL endpoint (CSV export)
- **Content:** Concepts for art, architecture, and related objects, including preferred labels and scope notes (definitions).

### 2. MIMO (Musical Instrument Museums Online)
- **Domain:** Musical Instruments
- **Source:** DOREMUS project (Turtle/RDF)
- **Format:** Turtle file, enriched with DBpedia and Wikidata descriptions
- **Content:** Instrument names, URIs, and multilingual labels. Definitions are supplemented from external sources.

### 3. PBCore
- **Domain:** Audiovisual Media
- **Source:** PBCore vocabulary spreadsheets
- **Format:** Excel (converted to CSV)
- **Content:** Asset types and instantiations for audiovisual objects, with terms and definitions.

### 4. Europeana Fashion
- **Domain:** Clothing and Fashion
- **Source:** Europeana Fashion RDF
- **Format:** RDF/XML (queried via SPARQL)
- **Content:** Fashion-related concepts, preferred labels, and scope notes.

---

## RAG-Based Classification Workflow

1. **Vocabulary Preparation**
   - Each vocabulary is parsed and normalized into a unified format: `term`, `definition`, `uri`, and `source_vocabulary`.
   - All vocabularies are merged into a single DataFrame and saved as `all_concepts.csv`.

2. **Vector Database Construction**
   - Terms and definitions are embedded using a SentenceTransformer model (e.g., `all-MiniLM-L6-v2`).
   - Embeddings are indexed with FAISS for fast similarity search.
   - Metadata is stored to map index positions to vocabulary entries.

3. **Retrieval**
   - Given an object description, its embedding is computed.
   - The FAISS index retrieves the top-k most similar vocabulary terms.

4. **Prompt Generation**
   - The retrieved candidates are formatted into a prompt for the LLM (e.g., Gemini).
   - The prompt includes the object description and numbered candidate terms.

5. **Classification**
   - The LLM selects the most appropriate term(s) by returning the number(s) corresponding to the candidates.
   - The result is mapped back to the vocabulary entry for explainable classification.

---

## Example Usage
The first step is to generate the vector database with the term embeddings from the different vocabularies. There should be a separate vector database for each vocabulary, so that they can be queried separately depending on the broad category of an object (e.g. you would query MIMO if an object is categorised as a musical instrument). Below, we provide an example to create a vector database for the whole vocabulary for simplicity.

```bash
python classification.py build --vocabulary vocabularies/all_concepts.csv --model_name all-MiniLM-L6-v2 --index_strategy term  --output_dir vocabularies/vectordb --vocabulary_name all_concepts
```

```python
from classification import RAGObjectClassifier

classifier = RAGObjectClassifier(
    index_path="vocabularies/vectordb/all_concepts_min.index",
    metadata_path="vocabularies/vectordb/metadata.pkl",
    generator_model='gemini-2.5-flash',
    embedding_model='all-MiniLM-L6-v2',
    generator_api_key="YOUR_GEMINI_API_KEY"
)

vocab_number, vocab_candidates = classifier.classify(
    description="The Blues Brother Gibson ES-335 guitar",
    classification_prompt='single'
)

print(f"Selected vocabulary number(s): {vocab_number}")
for i in vocab_number:
    print(f"Term: {vocab_candidates[i]['term']}, Source: {vocab_candidates[i]['source_vocabulary']}")
```

## Broad Categories matching

## In AAT...
Broad categories are mapped as follows:
- **Music Instrument" → *sound devices (equipment)* <http://vocab.getty.edu/aat/300387677>
- **Clothing** → *costume (mode of fashion)* <http://vocab.getty.edu/aat/300178802>
- **Audiovisual Media** → *audiovisual material* <http://vocab.getty.edu/aat/300028045>
- **Textual & Graphic Works** → *information forms (objects)* <http://vocab.getty.edu/aat/300220751>
- **Merchandise** → merchandise tags <http://vocab.getty.edu/aat/300427382>
- **Personal Effects** → *personalia* <http://vocab.getty.edu/aat/300259670>
- **Award & Recognition** → *honoraria (awards)* <http://vocab.getty.edu/aat/300380126>

If you believe that more informative terms should be reused for AAT, please reach out or open an issue!

## In Wikidata...

Broad categories are mapped as follows (Wikidata wiki links are provided for simplicity/navigation):
- **Music Instrument** as *music instrument* <https://www.wikidata.org/wiki/Q34379> or more generally *music equipment* <https://www.wikidata.org/wiki/Q56146933> or even more broadly, in the same spirit of AAT, *audio generator* <https://www.wikidata.org/wiki/Q267228>
- **Clothing** as *clothing (covering worn on the body)* <https://www.wikidata.org/wiki/Q11460>
- **Audiovisual** media as *audiovisual (electronic media with both a sound and a visual component, such as films and television programs)* <https://www.wikidata.org/wiki/Q758901>
- **Audiovisual Media** as *document (form for preservation of structured and identified information)* <https://www.wikidata.org/wiki/Q49848>
- **Merchandise** as *goods (tangible or intangible thing, except a service, that satisfies human wants and provides utility)* <https://www.wikidata.org/wiki/Q28877>
- **Personal Effects** as *personal effects (privately owned items (such as clothing, cosmetics and jewelry) normally worn or carried by a person*) <https://www.wikidata.org/wiki/Q115866017>
- **Award & Recognition** as  *award (something given to a person or a group of people to recognize their merit or excellence)* <https://www.wikidata.org/wiki/Q618779>

## Notes

* Each object has typically one main broad category, which is manually assigned (ideally at creation time) or assigned by an LLM validated on a testset (at the data lifting stage) based on the 7 possible choices (see above). Nonetheless, for certain broad categories (e.g. *Merchandise*, *Personal Effects*), an object/item may receive multiple broad categories in theory. For example, "John Lennon's glasses" is technically both *Clothing* and *Personal Effects* and "Beatles concert t-shirt" is both *Merchandise* and *Clothing*. This is rare, but it happens!
* For simplicity, in the KG API, an item/object is assigned to its broad category(ies) via [`scho:category`](https://schema.org/category). Similarly, the association to the specific category/type is achieved via [`scho:additionalType`](https://schema.org/additionalType).

Some terms cannot be found, and they should be added to AAT, such as:
- Flyers (e.g. Flyer for The Detours at the White Hart Hotel in Acton) → currently renamed to leaflets;
- Tour Programme (e.g. The Beatles tour programme) → currently minted anew;
- Lyrics (a significant type of object in a music museum) → currently reused from WD (Q602446,lyrics,sequence of words that make up a song);
- Photograph → reused from WD (Q125191,photograph,image created by light falling on a light-sensitive surface).

## Creating the tensor databases


```bash
python classification.py build \
    --vocabulary vocabularies/all_concepts.csv \
    --subset AAT \
    --model_name all-MiniLM-L6-v2 \
    --index_strategy term \
    --output_dir vocabularies/aat/vectordb \
    --vocabulary_name aat
 ```

 ```bash
python classification.py build \
    --vocabulary vocabularies/all_concepts.csv \
    --subset MIMO \
    --model_name all-MiniLM-L6-v2 \
    --index_strategy term \
    --output_dir vocabularies/mimo/vectordb \
    --vocabulary_name mimo
 ```

 ```bash
 python classification.py build \
    --vocabulary vocabularies/all_concepts.csv \
    --subset Europeana \
    --model_name all-MiniLM-L6-v2 \
    --index_strategy term \
    --output_dir vocabularies/fashion/vectordb \
    --vocabulary_name europeana
```

 ```bash
python classification.py build \
    --vocabulary vocabularies/all_concepts.csv \
    --subset PBCore-AssetType PBCore-Instantiation \
    --model_name all-MiniLM-L6-v2 \
    --index_strategy term \
    --output_dir vocabularies/pbcore/vectordb \
    --vocabulary_name pbcore
 ```

