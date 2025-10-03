import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta
import numpy as np


class RAGAnalyzer:
    """Analysis tools for RAG system performance and comparison"""

    def __init__(self, db_path: str = "rag_logs.db"):
        self.db_path = db_path

    def get_config_performance_comparison(self) -> pd.DataFrame:
        """Compare performance across different configurations"""
        query = """
        SELECT
            c.config_id,
            c.llm_provider,
            c.llm_model,
            c.embedding_provider,
            c.embedding_model,
            c.chunk_size,
            c.top_k,
            c.enhanced_processing,
            COUNT(q.query_id) as total_queries,
            AVG(q.response_time_seconds) as avg_response_time,
            MIN(q.response_time_seconds) as min_response_time,
            MAX(q.response_time_seconds) as max_response_time,
            AVG(COALESCE(uf.rating, 0)) as avg_user_rating,
            COUNT(uf.feedback_id) as total_feedback
        FROM configurations c
        LEFT JOIN queries q ON c.config_id = q.config_id
        LEFT JOIN responses r ON q.query_id = r.query_id
        LEFT JOIN user_feedback uf ON r.response_id = uf.response_id
        GROUP BY c.config_id
        HAVING total_queries > 0
        ORDER BY total_queries DESC, avg_response_time ASC
        """

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)

    def get_llm_provider_comparison(self) -> pd.DataFrame:
        """Compare performance between different LLM providers"""
        query = """
        SELECT
            c.llm_provider,
            COUNT(q.query_id) as total_queries,
            AVG(q.response_time_seconds) as avg_response_time,
            AVG(COALESCE(uf.rating, 0)) as avg_rating,
            COUNT(DISTINCT c.config_id) as configurations_tested
        FROM configurations c
        LEFT JOIN queries q ON c.config_id = q.config_id
        LEFT JOIN responses r ON q.query_id = r.query_id
        LEFT JOIN user_feedback uf ON r.response_id = uf.response_id
        GROUP BY c.llm_provider
        HAVING total_queries > 0
        ORDER BY avg_rating DESC, avg_response_time ASC
        """

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)

    def get_embedding_provider_comparison(self) -> pd.DataFrame:
        """Compare performance between different embedding providers"""
        query = """
        SELECT
            c.embedding_provider,
            c.embedding_model,
            COUNT(q.query_id) as total_queries,
            AVG(q.response_time_seconds) as avg_response_time,
            AVG(COALESCE(uf.rating, 0)) as avg_rating
        FROM configurations c
        LEFT JOIN queries q ON c.config_id = q.config_id
        LEFT JOIN responses r ON q.query_id = r.query_id
        LEFT JOIN user_feedback uf ON r.response_id = uf.response_id
        GROUP BY c.embedding_provider, c.embedding_model
        HAVING total_queries > 0
        ORDER BY avg_rating DESC
        """

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)

    def get_chunk_size_analysis(self) -> pd.DataFrame:
        """Analyze the impact of chunk size on performance"""
        query = """
        SELECT
            c.chunk_size,
            c.chunk_overlap,
            COUNT(q.query_id) as total_queries,
            AVG(q.response_time_seconds) as avg_response_time,
            AVG(q.retrieved_chunks_count) as avg_chunks_retrieved,
            AVG(COALESCE(uf.rating, 0)) as avg_rating
        FROM configurations c
        LEFT JOIN queries q ON c.config_id = q.config_id
        LEFT JOIN responses r ON q.query_id = r.query_id
        LEFT JOIN user_feedback uf ON r.response_id = uf.response_id
        GROUP BY c.chunk_size, c.chunk_overlap
        HAVING total_queries > 0
        ORDER BY c.chunk_size
        """

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)

    def get_enhanced_vs_basic_processing(self) -> pd.DataFrame:
        """Compare enhanced vs basic processing performance"""
        query = """
        SELECT
            c.enhanced_processing,
            COUNT(q.query_id) as total_queries,
            AVG(q.response_time_seconds) as avg_response_time,
            AVG(COALESCE(uf.rating, 0)) as avg_rating,
            AVG(q.retrieved_chunks_count) as avg_chunks_retrieved
        FROM configurations c
        LEFT JOIN queries q ON c.config_id = q.config_id
        LEFT JOIN responses r ON q.query_id = r.query_id
        LEFT JOIN user_feedback uf ON r.response_id = uf.response_id
        GROUP BY c.enhanced_processing
        HAVING total_queries > 0
        """

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)

    def get_query_patterns(self, limit: int = 20) -> pd.DataFrame:
        """Analyze common query patterns and performance"""
        query = """
        SELECT
            q.user_question,
            COUNT(*) as frequency,
            AVG(q.response_time_seconds) as avg_response_time,
            AVG(COALESCE(uf.rating, 0)) as avg_rating,
            MIN(q.asked_at) as first_asked,
            MAX(q.asked_at) as last_asked
        FROM queries q
        LEFT JOIN responses r ON q.query_id = r.query_id
        LEFT JOIN user_feedback uf ON r.response_id = uf.response_id
        GROUP BY q.query_hash
        HAVING frequency > 1
        ORDER BY frequency DESC
        LIMIT ?
        """

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=(limit,))

    def get_time_series_analysis(self, days: int = 30) -> pd.DataFrame:
        """Analyze usage patterns over time"""
        cutoff_date = datetime.now() - timedelta(days=days)

        query = """
        SELECT
            DATE(q.asked_at) as date,
            COUNT(q.query_id) as daily_queries,
            AVG(q.response_time_seconds) as avg_response_time,
            AVG(COALESCE(uf.rating, 0)) as avg_rating,
            COUNT(DISTINCT q.session_id) as unique_sessions
        FROM queries q
        LEFT JOIN responses r ON q.query_id = r.query_id
        LEFT JOIN user_feedback uf ON r.response_id = uf.response_id
        WHERE q.asked_at >= ?
        GROUP BY DATE(q.asked_at)
        ORDER BY date
        """

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=(cutoff_date.isoformat(),))

    def get_top_k_analysis(self) -> pd.DataFrame:
        """Analyze the impact of top_k parameter on performance"""
        query = """
        SELECT
            c.top_k,
            COUNT(q.query_id) as total_queries,
            AVG(q.response_time_seconds) as avg_response_time,
            AVG(q.retrieved_chunks_count) as avg_chunks_retrieved,
            AVG(COALESCE(uf.rating, 0)) as avg_rating
        FROM configurations c
        LEFT JOIN queries q ON c.config_id = q.config_id
        LEFT JOIN responses r ON q.query_id = r.query_id
        LEFT JOIN user_feedback uf ON r.response_id = uf.response_id
        GROUP BY c.top_k
        HAVING total_queries > 0
        ORDER BY c.top_k
        """

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)

    def plot_config_performance(self, save_path: str = None):
        """Create visualization of configuration performance"""
        df = self.get_config_performance_comparison()

        if df.empty:
            print("No data available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Response time vs rating
        axes[0, 0].scatter(df['avg_response_time'], df['avg_user_rating'],
                          s=df['total_queries']*10, alpha=0.6)
        axes[0, 0].set_xlabel('Average Response Time (seconds)')
        axes[0, 0].set_ylabel('Average User Rating')
        axes[0, 0].set_title('Response Time vs User Rating\n(Bubble size = Query count)')

        # LLM Provider comparison
        llm_data = df.groupby('llm_provider').agg({
            'avg_response_time': 'mean',
            'avg_user_rating': 'mean',
            'total_queries': 'sum'
        }).reset_index()

        axes[0, 1].bar(llm_data['llm_provider'], llm_data['avg_user_rating'])
        axes[0, 1].set_xlabel('LLM Provider')
        axes[0, 1].set_ylabel('Average User Rating')
        axes[0, 1].set_title('Average Rating by LLM Provider')

        # Chunk size vs performance
        axes[1, 0].scatter(df['chunk_size'], df['avg_response_time'],
                          c=df['avg_user_rating'], cmap='viridis', alpha=0.7)
        axes[1, 0].set_xlabel('Chunk Size')
        axes[1, 0].set_ylabel('Average Response Time')
        axes[1, 0].set_title('Chunk Size vs Response Time\n(Color = User Rating)')

        # Enhanced vs Basic processing
        enhanced_data = df.groupby('enhanced_processing').agg({
            'avg_response_time': 'mean',
            'avg_user_rating': 'mean'
        }).reset_index()

        x_pos = [0, 1]
        axes[1, 1].bar(x_pos, enhanced_data['avg_response_time'], alpha=0.7, label='Response Time')
        ax_twin = axes[1, 1].twinx()
        ax_twin.bar([x + 0.4 for x in x_pos], enhanced_data['avg_user_rating'],
                   alpha=0.7, color='orange', label='User Rating')
        axes[1, 1].set_xlabel('Enhanced Processing')
        axes[1, 1].set_ylabel('Response Time (seconds)')
        ax_twin.set_ylabel('User Rating')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(['Basic', 'Enhanced'])
        axes[1, 1].set_title('Basic vs Enhanced Processing')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_time_series(self, days: int = 30, save_path: str = None):
        """Plot usage and performance over time"""
        df = self.get_time_series_analysis(days)

        if df.empty:
            print("No data available for time series analysis")
            return

        df['date'] = pd.to_datetime(df['date'])

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Daily query count
        axes[0, 0].plot(df['date'], df['daily_queries'], marker='o')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Daily Queries')
        axes[0, 0].set_title('Daily Query Volume')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Response time trend
        axes[0, 1].plot(df['date'], df['avg_response_time'], marker='o', color='orange')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Average Response Time (seconds)')
        axes[0, 1].set_title('Response Time Trend')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # User rating trend
        axes[1, 0].plot(df['date'], df['avg_rating'], marker='o', color='green')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Average User Rating')
        axes[1, 0].set_title('User Rating Trend')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Unique sessions
        axes[1, 1].plot(df['date'], df['unique_sessions'], marker='o', color='red')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Unique Sessions')
        axes[1, 1].set_title('Daily Active Sessions')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'config_comparison': self.get_config_performance_comparison().to_dict('records'),
            'llm_comparison': self.get_llm_provider_comparison().to_dict('records'),
            'embedding_comparison': self.get_embedding_provider_comparison().to_dict('records'),
            'chunk_size_analysis': self.get_chunk_size_analysis().to_dict('records'),
            'enhanced_vs_basic': self.get_enhanced_vs_basic_processing().to_dict('records'),
            'top_queries': self.get_query_patterns().to_dict('records'),
            'time_series': self.get_time_series_analysis().to_dict('records')
        }

        return report

    def save_report(self, output_file: str = None) -> str:
        """Save performance report to JSON file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"rag_performance_report_{timestamp}.json"

        report = self.generate_performance_report()

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        return output_file

    def get_best_configurations(self, metric: str = 'avg_user_rating', top_n: int = 5) -> pd.DataFrame:
        """Get the best performing configurations based on a specific metric"""
        df = self.get_config_performance_comparison()

        if df.empty:
            return df

        # Filter out configs with very few queries (less than 5)
        df_filtered = df[df['total_queries'] >= 5]

        if df_filtered.empty:
            df_filtered = df

        return df_filtered.nlargest(top_n, metric)

    def compare_two_configs(self, config_id1: str, config_id2: str) -> Dict[str, Any]:
        """Detailed comparison between two specific configurations"""
        query = """
        SELECT
            c.config_id,
            c.llm_provider,
            c.llm_model,
            c.embedding_provider,
            c.embedding_model,
            c.chunk_size,
            c.top_k,
            c.enhanced_processing,
            COUNT(q.query_id) as total_queries,
            AVG(q.response_time_seconds) as avg_response_time,
            MIN(q.response_time_seconds) as min_response_time,
            MAX(q.response_time_seconds) as max_response_time,
            AVG(COALESCE(uf.rating, 0)) as avg_user_rating,
            COUNT(uf.feedback_id) as total_feedback
        FROM configurations c
        LEFT JOIN queries q ON c.config_id = q.config_id
        LEFT JOIN responses r ON q.query_id = r.query_id
        LEFT JOIN user_feedback uf ON r.response_id = uf.response_id
        WHERE c.config_id IN (?, ?)
        GROUP BY c.config_id
        """

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=(config_id1, config_id2))

        if df.empty or len(df) < 2:
            return {"error": "Could not find both configurations"}

        config1 = df.iloc[0].to_dict()
        config2 = df.iloc[1].to_dict()

        comparison = {
            'config1': config1,
            'config2': config2,
            'performance_diff': {
                'response_time_diff': config2['avg_response_time'] - config1['avg_response_time'],
                'rating_diff': config2['avg_user_rating'] - config1['avg_user_rating'],
                'query_count_diff': config2['total_queries'] - config1['total_queries']
            },
            'winner': {
                'faster': config1['config_id'] if config1['avg_response_time'] < config2['avg_response_time'] else config2['config_id'],
                'higher_rated': config1['config_id'] if config1['avg_user_rating'] > config2['avg_user_rating'] else config2['config_id'],
                'more_tested': config1['config_id'] if config1['total_queries'] > config2['total_queries'] else config2['config_id']
            }
        }

        return comparison


# Example usage
def example_analysis():
    """Example of how to use the RAGAnalyzer"""
    analyzer = RAGAnalyzer()

    # Get performance comparison
    config_perf = analyzer.get_config_performance_comparison()
    print("Configuration Performance:")
    print(config_perf.head())

    # Compare LLM providers
    llm_comp = analyzer.get_llm_provider_comparison()
    print("\nLLM Provider Comparison:")
    print(llm_comp)

    # Get best configurations
    best_configs = analyzer.get_best_configurations(metric='avg_user_rating', top_n=3)
    print("\nBest Configurations by User Rating:")
    print(best_configs)

    # Generate plots
    analyzer.plot_config_performance(save_path='config_performance.png')
    analyzer.plot_time_series(days=30, save_path='time_series.png')

    # Save comprehensive report
    report_file = analyzer.save_report()
    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    example_analysis()