from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
import logging
from enum import Enum, auto
import sys
from typing import Any, Generic, List, Optional, Tuple, TypeVar, Union
from langchain_core.documents import Document as LCDocument
from langchain_core.vectorstores import VectorStore as LCVectorStore
from paperqa.types import Text as PQAText
from paperqa.types import Doc as PQADoc

from src.storage.vector.errors import PipelineError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
# Generic type variables for source and target document types
SourceDoc = TypeVar("SourceDoc")
TargetDoc = TypeVar("TargetDoc")


class TextStorageType(Enum):
    """Enum for supported document types."""

    LANGCHAIN = auto()
    PAPERQA = auto()
    # Add other document types here as needed


@dataclass
class ConversionResult(Generic[TargetDoc]):
    """Generic result of document conversion."""

    successful: List[TargetDoc]
    failed: List[Any]  # Original documents that failed conversion
    errors: List[str]


class DocumentAdapter(ABC, Generic[SourceDoc, TargetDoc]):
    """Abstract base class for document adapters."""

    @abstractmethod
    def convert(self, doc: SourceDoc) -> TargetDoc:
        """Convert a single document from source to target type."""
        pass

    @abstractmethod
    def convert_back(self, doc: TargetDoc) -> SourceDoc:
        """Convert a single document from target back to source type."""
        pass

    def batch_convert(self, docs: List[SourceDoc]) -> ConversionResult[TargetDoc]:
        """Convert a batch of documents from source to target type."""
        successful = []
        failed = []
        errors = []

        for doc in docs:
            try:
                converted = self.convert(doc)
                successful.append(converted)
            except Exception as e:
                failed.append(doc)
                errors.append(f"Conversion failed: {str(e)}")

        return ConversionResult(successful, failed, errors)

    def batch_convert_back(self, docs: List[TargetDoc]) -> ConversionResult[SourceDoc]:
        """Convert a batch of documents from target back to source type."""
        successful = []
        failed = []
        errors = []

        for doc in docs:
            try:
                converted = self.convert_back(doc)
                successful.append(converted)
            except Exception as e:
                failed.append(doc)
                errors.append(f"Conversion failed: {str(e)}")

        return ConversionResult(successful, failed, errors)


class LangChainPaperQAAdapter(DocumentAdapter[LCDocument, PQAText]):
    """Adapter for converting between LangChain and PaperQA documents."""

    def convert(self, doc: LCDocument) -> PQAText:
        """Convert LangChain Document to PaperQA Text."""
        pqa_doc = PQADoc(
            docname=doc.metadata.get("name", doc.metadata.get("id", "")),
            citation=doc.metadata.get("citation", ""),
            dockey=doc.metadata.get("id", ""),
        )

        return PQAText(
            text=doc.page_content,
            name=doc.metadata.get("name", doc.metadata.get("id", "")),
            doc=pqa_doc,
            embedding=doc.metadata.get("embedding", None),
        )

    def convert_back(self, doc: PQAText) -> LCDocument:
        """Convert PaperQA Text to LangChain Document."""
        metadata = {
            "name": doc.name,
            "id": doc.doc.dockey if doc.doc else "",
            "citation": doc.doc.citation if doc.doc else "",
        }

        if doc.embedding is not None:
            metadata["embedding"] = doc.embedding

        return LCDocument(page_content=doc.text, metadata=metadata)


class AdapterFactory:
    """Factory for creating document adapters."""

    _adapters = {
        (TextStorageType.LANGCHAIN, TextStorageType.PAPERQA): LangChainPaperQAAdapter,
        # Add more adapter mappings here
    }

    @classmethod
    def get_adapter(
        cls, source_type: TextStorageType, target_type: TextStorageType
    ) -> DocumentAdapter:
        """Get appropriate adapter for the given source and target types."""
        if source_type == target_type:
            logger.info(
                f"No adapter needed for conversion from {source_type} to {target_type}"
            )
            return
        adapter_class = cls._adapters.get((source_type, target_type))
        if adapter_class is None:
            raise ValueError(
                f"No adapter found for conversion from {source_type} to {target_type}"
            )
        return adapter_class()

    @classmethod
    def register_adapter(
        cls,
        source_type: TextStorageType,
        target_type: TextStorageType,
        adapter_class: type[DocumentAdapter],
    ) -> None:
        """Register a new adapter for document conversion."""
        cls._adapters[(source_type, target_type)] = adapter_class


class BaseVectorStorePipeline(ABC):
    """Contract for a vector store pipelines to handle vector store operations and
    object type conversions.
    """

    @abstractmethod
    def add_texts(self, texts: List[str], **kwargs: Any) -> Tuple[int, List[str]]:
        """Add text embeddings to the vector store. Intended simply wrap the base add
        text method of the vector store. Subclasses may require additional
        parameters if metadata is needed.
        """
        pass

    @abstractmethod
    def add_documents(
        self, docs: List[Any], source_type: TextStorageType, **kwargs: Any
    ) -> Tuple[int, List[str]]:
        """Add documents of any supported type to the vector store. Adapters may be
        needed to convert documents to the target vector store type.
        """
        pass

    @abstractmethod
    async def search(
        self, query_embedding: List[float], k: int, search_type: str, **kwargs: Any
    ) -> List[Any]:
        """Search the vector store and convert results to target document type."""
        pass

    @property
    @abstractmethod
    def vector_store(self) -> Any:
        """Get the vector store class."""
        pass


class LCVectorStorePipeline(BaseVectorStorePipeline):
    """Generic wrapper for a LangChain vector store to handle document type conversion."""

    def __init__(
        self,
        vector_store: Optional[LCVectorStore] = None,
        target_type: TextStorageType = TextStorageType.LANGCHAIN,
    ):
        """Initialize wrapper with vector store and target document type."""
        self._vector_store = vector_store
        self.target_type = target_type
        self.adapter = AdapterFactory.get_adapter(
            TextStorageType.LANGCHAIN, target_type
        )

    @property
    def vector_store(self) -> Any:
        """Get the vector store class."""
        return self._vector_store

    async def add_documents(
        self, docs: List[Any], source_type: TextStorageType
    ) -> Tuple[int, List[str]]:
        """Add documents of any supported type to the vector store."""
        self._ensure_vector_store()
        # LangChain documents need to conversion.
        if source_type == TextStorageType.LANGCHAIN:
            try:
                await self.vector_store.add_documents(docs)
                return len(docs), []
            except Exception as e:
                return 0, [str(e)]

        adapter = AdapterFactory.get_adapter(source_type, TextStorageType.LANGCHAIN)
        conversion = adapter.batch_convert_back(docs)

        if conversion.failed:
            return len(conversion.successful), conversion.errors

        try:
            await self.vector_store.add_documents(conversion.successful)
            return len(docs), []
        except Exception as e:
            return 0, [f"Failed to add documents to vector store: {str(e)}"]

    async def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Tuple[int, List[str]]:
        """Add texts to the vector store."""
        self._ensure_vector_store()
        return await self.vector_store.add_texts(
            texts=texts, metadatas=metadatas, ids=ids, **kwargs
        )

    async def search(
        self, query_embedding: List[float], k: int, search_type: str
    ) -> List[Any]:
        """Search vector store and convert results to target document type."""
        self._ensure_vector_store()
        results = await self.vector_store.search(query_embedding, k, search_type)

        if self.target_type == TextStorageType.LANGCHAIN:
            return results

        conversion = self.adapter.batch_convert(results)
        if conversion.failed:
            for error in conversion.errors:
                print(f"Warning: {error}")

        return conversion.successful

    def _ensure_vector_store(self) -> LCVectorStore:
        """Ensure the vector store is set. If not set, cease execution."""
        if not self.vector_store:
            raise PipelineError("Vector store not set. Cannot continue execution.")


# Example of adding a new document type and adapter
class CustomDoc:
    """Example custom document type."""

    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata


class CustomDocAdapter(DocumentAdapter[LCDocument, CustomDoc]):
    """Example adapter for custom document type."""

    def convert(self, doc: LCDocument) -> CustomDoc:
        return CustomDoc(content=doc.page_content, metadata=doc.metadata)

    def convert_back(self, doc: CustomDoc) -> LCDocument:
        return LCDocument(page_content=doc.content, metadata=doc.metadata)


# Usage example
async def main():
    # Register new document type and adapter
    class CustomDocType(TextStorageType):
        CUSTOM = auto()

    AdapterFactory.register_adapter(
        TextStorageType.LANGCHAIN, CustomDocType.CUSTOM, CustomDocAdapter
    )

    # Initialize vector store wrapper
    vector_store = LCVectorStore(...)
    wrapper = LCVectorStorePipeline(vector_store, target_type=CustomDocType.CUSTOM)

    # Add documents
    custom_docs = [CustomDoc(...)]
    added, errors = await wrapper.add_documents(
        custom_docs, source_type=CustomDocType.CUSTOM
    )

    # Search
    results = await wrapper.search(
        query_embedding=[...], k=5, use_mmr=True
    )  # Results will be CustomDoc objects
