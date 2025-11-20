from typing import List, Optional
from pathlib import Path
import arxiv

from .base import BaseScraper, PaperMetadata, Author
from .exceptions import InvalidQueryError, DownloadError, APIError
from .utils import RateLimiter, SimpleCache, sanitize_filename, retry_on_failure, create_output_directory
from .. import ScraperSourceConfig


class ArxivScraper(BaseScraper):

    CATEGORIES = {
        # "Computer Science"
        "cs.AI": "Artificial Intelligence",
        "cs.AR": "Hardware Architecture",
        "cs.CC": "Computational Complexity",
        "cs.CE": "Computational Engineering, Finance, and Science",
        "cs.CG": "Computational Geometry",
        "cs.CL": "Computation and Language",
        "cs.CR": "Cryptography and Security",
        "cs.CV": "Computer Vision and Pattern Recognition",
        "cs.CY": "Computers and Society",
        "cs.DB": "Databases",
        "cs.DC": "Distributed, Parallel, and Cluster Computing",
        "cs.DL": "Digital Libraries",
        "cs.DM": "Discrete Mathematics",
        "cs.DS": "Data Structures and Algorithms",
        "cs.ET": "Emerging Technologies",
        "cs.FL": "Formal Languages and Automata Theory",
        "cs.GL": "General Literature",
        "cs.GR": "Graphics",
        "cs.GT": "Computer Science and Game Theory",
        "cs.HC": "Human-Computer Interaction",
        "cs.IR": "Information Retrieval",
        "cs.IT": "Information Theory",
        "cs.LG": "Machine Learning",
        "cs.LO": "Logic in Computer Science",
        "cs.MA": "Multiagent Systems",
        "cs.MM": "Multimedia",
        "cs.MS": "Mathematical Software",
        "cs.NA": "Numerical Analysis",
        "cs.NE": "Neural and Evolutionary Computing",
        "cs.NI": "Networking and Internet Architecture",
        "cs.OH": "Other Computer Science",
        "cs.OS": "Operating Systems",
        "cs.PF": "Performance",
        "cs.PL": "Programming Languages",
        "cs.RO": "Robotics",
        "cs.SC": "Symbolic Computation",
        "cs.SD": "Sound",
        "cs.SE": "Software Engineering",
        "cs.SI": "Social and Information Networks",
        "cs.SY": "Systems and Control",
        "econ.EM": "Econometrics",
        "econ.GN": "General Economics",
        "econ.TH": "Theoretical Economics",
        # "Electrical Engineering and Systems Science"
        "eess.AS": "Audio and Speech Processing",
        "eess.IV": "Image and Video Processing",
        "eess.SP": "Signal Processing",
        "eess.SY": "Systems and Control",
        # "Mathematics"
        "math.AC": "Commutative Algebra",
        "math.AG": "Algebraic Geometry",
        "math.AP": "Analysis of PDEs",
        "math.AT": "Algebraic Topology",
        "math.CA": "Classical Analysis and ODEs",
        "math.CO": "Combinatorics",
        "math.CT": "Category Theory",
        "math.CV": "Complex Variables",
        "math.DG": "Differential Geometry",
        "math.DS": "Dynamical Systems",
        "math.FA": "Functional Analysis",
        "math.GM": "General Mathematics",
        "math.GN": "General Topology",
        "math.GR": "Group Theory",
        "math.GT": "Geometric Topology",
        "math.HO": "History and Overview",
        "math.IT": "Information Theory (alias cs.IT)",
        "math.KT": "K-Theory and Homology",
        "math.LO": "Logic",
        "math.MG": "Metric Geometry",
        "math.MP": "Mathematical Physics (alias math-ph)",
        "math.NA": "Numerical Analysis",
        "math.NT": "Number Theory",
        "math.OA": "Operator Algebras",
        "math.OC": "Optimization and Control",
        "math.PR": "Probability",
        "math.QA": "Quantum Algebra",
        "math.RA": "Rings and Algebras",
        "math.RT": "Representation Theory",
        "math.SG": "Symplectic Geometry",
        "math.SP": "Spectral Theory",
        "math.ST": "Statistics Theory",
        # "Physics"
        "astro-ph.CO": "Cosmology and Nongalactic Astrophysics",
        "astro-ph.EP": "Earth and Planetary Astrophysics",
        "astro-ph.GA": "Astrophysics of Galaxies",
        "astro-ph.HE": "High Energy Astrophysical Phenomena",
        "astro-ph.IM": "Instrumentation and Methods for Astrophysics",
        "astro-ph.SR": "Solar and Stellar Astrophysics",
        "cond-mat.dis-nn": "Disordered Systems and Neural Networks",
        "cond-mat.mes-hall": "Mesoscale and Nanoscale Physics",
        "cond-mat.mtrl-sci": "Materials Science",
        "cond-mat.other": "Other Condensed Matter",
        "cond-mat.quant-gas": "Quantum Gases",
        "cond-mat.soft": "Soft Condensed Matter",
        "cond-mat.stat-mech": "Statistical Mechanics",
        "cond-mat.str-el": "Strongly Correlated Electrons",
        "cond-mat.supr-con": "Superconductivity",
        "gr-qc": "General Relativity and Quantum Cosmology",
        "hep-ex": "High Energy Physics - Experiment",
        "hep-lat": "High Energy Physics - Lattice",
        "hep-ph": "High Energy Physics - Phenomenology",
        "hep-th": "High Energy Physics - Theory",
        "math-ph": "Mathematical Physics",
        "nlin.AO": "Adaptation and Self-Organizing Systems",
        "nlin.CD": "Chaotic Dynamics",
        "nlin.CG": "Cellular Automata and Lattice Gases",
        "nlin.PS": "Pattern Formation and Solitons",
        "nlin.SI": "Exactly Solvable and Integrable Systems",
        "nucl-ex": "Nuclear Experiment",
        "nucl-th": "Nuclear Theory",
        "physics.acc-ph": "Accelerator Physics",
        "physics.ao-ph": "Atmospheric and Oceanic Physics",
        "physics.app-ph": "Applied Physics",
        "physics.atm-clus": "Atomic and Molecular Clusters",
        "physics.atom-ph": "Atomic Physics",
        "physics.bio-ph": "Biological Physics",
        "physics.chem-ph": "Chemical Physics",
        "physics.class-ph": "Classical Physics",
        "physics.comp-ph": "Computational Physics",
        "physics.data-an": "Data Analysis, Statistics and Probability",
        "physics.ed-ph": "Physics Education",
        "physics.flu-dyn": "Fluid Dynamics",
        "physics.gen-ph": "General Physics",
        "physics.geo-ph": "Geophysics",
        "physics.hist-ph": "History and Philosophy of Physics",
        "physics.ins-det": "Instrumentation and Detectors",
        "physics.med-ph": "Medical Physics",
        "physics.optics": "Optics",
        "physics.plasm-ph": "Plasma Physics",
        "physics.pop-ph": "Popular Physics",
        "physics.soc-ph": "Physics and Society",
        "physics.space-ph": "Space Physics",
        "quant-ph": "Quantum Physics",
        # "Quantitative Biology"
        "q-bio.BM": "Biomolecules",
        "q-bio.CB": "Cell Behavior",
        "q-bio.GN": "Genomics",
        "q-bio.MN": "Molecular Networks",
        "q-bio.NC": "Neurons and Cognition",
        "q-bio.OT": "Other Quantitative Biology",
        "q-bio.PE": "Populations and Evolution",
        "q-bio.QM": "Quantitative Methods",
        "q-bio.SC": "Subcellular Processes",
        "q-bio.TO": "Tissues and Organs",
        # "Quantitative Finance"
        "q-fin.CP": "Computational Finance",
        "q-fin.EC": "Economics (alias econ.GN)",
        "q-fin.GN": "General Finance",
        "q-fin.MF": "Mathematical Finance",
        "q-fin.PM": "Portfolio Management",
        "q-fin.PR": "Pricing of Securities",
        "q-fin.RM": "Risk Management",
        "q-fin.ST": "Statistical Finance",
        "q-fin.TR": "Trading and Market Microstructure",
        # "Statistics"
        "stat.AP": "Applications",
        "stat.CO": "Computation",
        "stat.ME": "Methodology",
        "stat.ML": "Machine Learning",
        "stat.OT": "Other Statistics",
        "stat.TH": "Statistics Theory (alias math.ST)"
    }

    def __init__(self, config: ScraperSourceConfig):
        super().__init__(config)
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=3,
            num_retries=config.retry_attempts
        )
        self.rate_limiter = RateLimiter(
            requests_per_period=config.rate_limit_requests,
            period_seconds=config.rate_limit_period
        )
        if config.cache_ttl > 0:
            self.cache = SimpleCache(ttl=config.cache_ttl)
        else:
            self.cache = None

    def _get_name(self) -> str:
        return "arxiv"

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        **filters
    ) -> List[PaperMetadata]:
        if not self.validate_query(query):
            raise InvalidQueryError(query, "Query cannot be empty")

        cache_key = f"{query}:{max_results}:{filters}"
        if self.cache:
            cached_results = self.cache.get(cache_key)
            if cached_results:
                return cached_results

        query_parts = []

        query_parts.append(f"all:{query}")

        categories = filters.get('categories', [])
        if categories:
            cat_query = " OR ".join([f'cat:{cat}' for cat in categories])
            query_parts.append(f"({cat_query})")

        date_from = filters.get('date_from')
        date_to = filters.get('date_to')
        if date_from or date_to:
            date_from_str = date_from.replace('-', '') + '0000' if date_from else '19910101000'
            date_to_str = date_to.replace('-', '') + '2359' if date_to else '99991231235'
            query_parts.append(f"submittedDate:[{date_from_str} TO {date_to_str}]")

        final_query = " AND ".join(query_parts)

        sort_by = filters.get('sort_by', 'relevance')
        sort_criterion_map = {
            'relevance': arxiv.SortCriterion.Relevance,
            'lastUpdatedDate': arxiv.SortCriterion.LastUpdatedDate,
            'submittedDate': arxiv.SortCriterion.SubmittedDate,
        }
        sort_criterion = sort_criterion_map.get(sort_by, arxiv.SortCriterion.Relevance)

        if max_results is None:
            max_results = self.config.max_results

        self.rate_limiter.acquire(self.name)

        try:
            search = arxiv.Search(
                query=final_query,
                max_results=max_results,
                sort_by=sort_criterion
            )

            results: List[PaperMetadata] = []
            for result in self.client.results(search):
                paper = self._convert_to_paper_metadata(result)
                results.append(paper)

            if self.cache:
                self.cache.set(cache_key, results)

            return results

        except Exception as e:
            raise APIError(self.name, message=str(e))

    @retry_on_failure(max_attempts=3, delay=1.0, backoff=2.0)
    def download_paper(
        self,
        paper: PaperMetadata,
        output_dir: str
    ) -> Path:
        if not paper.pdf_url:
            raise DownloadError(paper.id, self.name, "No PDF URL available")

        output_path = create_output_directory(output_dir, self.name)

        filename = sanitize_filename(f"{paper.id}_{paper.title[:50]}.pdf")
        file_path = output_path / filename

        if file_path.exists():
            return file_path

        # Rate limiting
        self.rate_limiter.acquire(self.name)

        try:
            # paper.id is already a clean arxiv ID (e.g., '1706.03762')
            search = arxiv.Search(id_list=[paper.id])
            result = next(self.client.results(search))

            result.download_pdf(dirpath=str(output_path), filename=filename)

            return file_path

        except Exception as e:
            raise DownloadError(paper.id, self.name, str(e))

    def get_paper_metadata(self, paper_id: str) -> PaperMetadata:
        # Clean paper ID
        paper_id = paper_id.replace('arxiv:', '').replace('v', '.')

        # Rate limiting
        self.rate_limiter.acquire(self.name)

        try:
            search = arxiv.Search(id_list=[paper_id])
            result = next(self.client.results(search))
            return self._convert_to_paper_metadata(result)
        except StopIteration:
            raise APIError(self.name, message=f"Paper {paper_id} not found")
        except Exception as e:
            raise APIError(self.name, message=str(e))

    def get_supported_categories(self) -> List[str]:
        return list(self.CATEGORIES.keys())

    def _convert_to_paper_metadata(self, result: arxiv.Result) -> PaperMetadata:
        authors = [
            Author(name=author.name)
            for author in result.authors
        ]

        # Extract categories
        categories = result.categories

        # Extract ID from URL and remove version suffix (e.g., "1706.03762v7" -> "1706.03762")
        paper_id = result.entry_id.split('/')[-1]
        # Remove version suffix if present (format: vN where N is version number)
        if 'v' in paper_id:
            paper_id = paper_id.rsplit('v', 1)[0]

        return PaperMetadata(
            id=paper_id,
            source=self.name,
            title=result.title,
            authors=authors,
            abstract=result.summary,
            published_date=result.published,
            updated_date=result.updated,
            pdf_url=result.pdf_url,
            html_url=result.entry_id,
            doi=result.doi,
            categories=categories,
            extra_metadata={
                'primary_category': result.primary_category,
                'comment': result.comment,
                'journal_ref': result.journal_ref,
                'links': [link.href for link in result.links]
            }
        )