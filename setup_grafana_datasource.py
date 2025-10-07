#!/usr/bin/env python3
"""
Grafana Datasource and Dashboard Setup for RAG Performance Monitoring
This script automatically configures Grafana with your RAG database and creates dashboards.
"""

import requests
import json
import os
import time
import sqlite3
from pathlib import Path

# Grafana configuration
GRAFANA_URL = "http://localhost:3000"
GRAFANA_USER = "admin"
GRAFANA_PASS = "admin"  # Change after first login


def wait_for_grafana():
    """Wait for Grafana to be ready"""
    print("⏳ Waiting for Grafana to start...")
    for i in range(30):
        try:
            response = requests.get(f"{GRAFANA_URL}/api/health", timeout=5)
            if response.status_code == 200:
                print("✅ Grafana is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    return False


def create_datasource():
    """Create SQLite datasource in Grafana"""
    print("📊 Creating SQLite datasource...")

    # For Docker container, use the mounted path
    db_path = "/var/lib/grafana/rag_logs.db"
    local_db_path = os.path.abspath("rag_logs.db")

    # Check if local database exists
    if not os.path.exists(local_db_path):
        print(f"❌ Database not found at {local_db_path}")
        print("💡 Make sure to run your RAG system first to create the database!")
        return False

    # First, try to delete existing datasource
    try:
        delete_response = requests.delete(
            f"{GRAFANA_URL}/api/datasources/name/RAG_SQLite",
            auth=(GRAFANA_USER, GRAFANA_PASS)
        )
        if delete_response.status_code == 200:
            print("🗑️ Deleted existing datasource")
    except:
        pass  # Ignore if doesn't exist

    datasource_config = {
        "name": "RAG_SQLite",
        "type": "frser-sqlite-datasource",
        "url": "",
        "access": "proxy",
        "isDefault": True,
        "jsonData": {
            "path": db_path
        },
        "uid": "RAG_SQLite"
    }

    try:
        print(f"🔍 Sending datasource config: {json.dumps(datasource_config, indent=2)}")
        response = requests.post(
            f"{GRAFANA_URL}/api/datasources",
            json=datasource_config,
            auth=(GRAFANA_USER, GRAFANA_PASS),
            headers={"Content-Type": "application/json"}
        )

        print(f"🔍 Datasource API Response:")
        print(f"   Status Code: {response.status_code}")
        print(f"   Headers: {dict(response.headers)}")
        print(f"   Response Body: {response.text}")

        if response.status_code == 200:
            response_data = response.json()
            print(f"🔍 Created datasource with ID: {response_data.get('id', 'Unknown')}")
            print(f"🔍 Created datasource with UID: {response_data.get('uid', 'Unknown')}")
            print("✅ SQLite datasource created successfully!")
            return True
        else:
            print(f"❌ Failed to create datasource: {response.status_code}")
            try:
                error_data = response.json()
                print(f"🔍 Error details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"🔍 Raw error response: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to Grafana: {e}")
        return False

def create_dashboard():
    """Create RAG performance dashboard"""
    print("📈 Creating RAG Performance Dashboard...")

    # First, let's verify the datasource exists and test a query
    try:
        datasources_response = requests.get(
            f"{GRAFANA_URL}/api/datasources",
            auth=(GRAFANA_USER, GRAFANA_PASS)
        )
        print(f"🔍 Available datasources: {datasources_response.status_code}")
        if datasources_response.status_code == 200:
            datasources = datasources_response.json()
            sqlite_ds = [ds for ds in datasources if ds['type'] == 'frser-sqlite-datasource']
            print(f"🔍 SQLite datasources found: {len(sqlite_ds)}")
            if sqlite_ds:
                print(f"🔍 SQLite datasource UID: {sqlite_ds[0].get('uid', 'No UID')}")
                actual_uid = sqlite_ds[0].get('uid', 'RAG_SQLite')
            else:
                print("❌ No SQLite datasource found!")
                actual_uid = 'RAG_SQLite'
        else:
            print(f"❌ Failed to get datasources: {datasources_response.text}")
            actual_uid = 'RAG_SQLite'
    except Exception as e:
        print(f"❌ Error checking datasources: {e}")
        actual_uid = 'RAG_SQLite'

    def sql_panel(panel_id, title, sql, panel_type="stat", grid=None, unit=None):
        panel = {
            "id": panel_id,
            "title": title,
            "type": panel_type,
            "gridPos": grid or {"h": 8, "w": 6, "x": 0, "y": 0},
            "targets": [
                {
                    "refId": "A",
                    "rawSql": sql.strip(),
                    "format": "table" if panel_type == "table" else "time_series" if panel_type == "timeseries" else "table",
                    "datasource": {
                        "type": "frser-sqlite-datasource",
                        "uid": actual_uid
                    }
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": 0},
                            {"color": "yellow", "value": 50},
                            {"color": "red", "value": 100}
                        ]
                    }
                }
            }
        }

        if unit:
            panel["fieldConfig"]["defaults"]["unit"] = unit

        print(f"🔍 Created panel: {title} with datasource UID: {actual_uid}")
        return panel

    dashboard_config = {
        "dashboard": {
            "id": None,
            "title": "RAG Performance Monitor",
            "tags": ["rag", "performance"],
            "timezone": "browser",
            "refresh": "30s",
            "time": {"from": "now-24h", "to": "now"},
            "panels": [
                # Stat panels
                sql_panel(1, "Total Queries",
                    "SELECT COUNT(*) as value FROM queries",
                    "stat", {"h": 8, "w": 6, "x": 0, "y": 0}
                ),
                sql_panel(2, "Average Response Time",
                    "SELECT AVG(response_time_seconds) as value FROM queries WHERE response_time_seconds IS NOT NULL",
                    "stat", {"h": 8, "w": 6, "x": 6, "y": 0}, unit="s"
                ),
                sql_panel(3, "Unique Configurations",
                    "SELECT COUNT(DISTINCT config_id) as value FROM configurations",
                    "stat", {"h": 8, "w": 6, "x": 12, "y": 0}
                ),
                sql_panel(4, "Active Configurations",
                    "SELECT COUNT(DISTINCT config_id) as value FROM queries WHERE asked_at >= datetime('now', '-24 hours')",
                    "stat", {"h": 8, "w": 6, "x": 18, "y": 0}
                ),
                # Timeseries panels (FIXED - removed Grafana macros)
                sql_panel(5, "Response Time Over Time",
                          """
                          SELECT 
                            strftime('%s', substr(asked_at, 1, 13) || ':00:00') as time,
                            COUNT(*) as value
                            FROM queries
                            WHERE asked_at >= datetime('now', '-24 hours')
                            GROUP BY substr(asked_at, 1, 13)
                            ORDER BY time
                          """
                    "timeseries", {"h": 8, "w": 12, "x": 0, "y": 8}, unit="s"
                ),
                sql_panel(6, "Queries Per Hour",
                    "SELECT datetime(substr(asked_at, 1, 14) || '00:00') as time, COUNT(*) as value FROM queries WHERE asked_at >= datetime('now', '-24 hours') GROUP BY datetime(substr(asked_at, 1, 14) || '00:00') ORDER BY time",
                    "timeseries", {"h": 8, "w": 12, "x": 12, "y": 8}
                ),
                # Table panels - LLM & Model Comparison
                sql_panel(7, "LLM Provider Performance",
                    """
                    SELECT c.llm_provider as 'Provider', c.llm_model as 'Model',
                           COUNT(q.query_id) as 'Queries',
                           ROUND(AVG(q.response_time_seconds), 3) as 'Avg Time (s)',
                           ROUND(MIN(q.response_time_seconds), 3) as 'Min Time (s)',
                           ROUND(MAX(q.response_time_seconds), 3) as 'Max Time (s)'
                    FROM configurations c
                    LEFT JOIN queries q ON c.config_id = q.config_id
                    GROUP BY c.llm_provider, c.llm_model
                    HAVING COUNT(q.query_id) > 0
                    ORDER BY 'Avg Time (s)' ASC
                    """,
                    "table", {"h": 8, "w": 12, "x": 0, "y": 16}
                ),
                sql_panel(8, "Embedding Model Comparison",
                    """
                    SELECT c.embedding_model as 'Embedding Model',
                           COUNT(q.query_id) as 'Queries',
                           ROUND(AVG(q.response_time_seconds), 3) as 'Avg Time (s)',
                           COUNT(DISTINCT c.llm_provider) as 'LLM Providers Used'
                    FROM configurations c
                    LEFT JOIN queries q ON c.config_id = q.config_id
                    GROUP BY c.embedding_model
                    HAVING COUNT(q.query_id) > 0
                    ORDER BY 'Avg Time (s)' ASC
                    """,
                    "table", {"h": 8, "w": 12, "x": 12, "y": 16}
                ),
                # Vector DB & Configuration Analysis
                sql_panel(9, "Vector Database Performance",
                    """
                    SELECT c.vector_db as 'Vector DB',
                   COUNT(q.query_id) as 'Queries',
                   ROUND(AVG(q.response_time_seconds), 3) as 'Avg Time (s)',
                   COUNT(DISTINCT c.embedding_model) as 'Embedding Models',
                   ROUND(AVG(c.chunk_size), 0) as 'Avg Chunk Size'
                   FROM configurations c
                   LEFT JOIN queries q ON c.config_id = q.config_id
                   GROUP BY c.vector_db   
                   HAVING COUNT(q.query_id) > 0
                   ORDER BY 'Avg Time (s)' ASC
                    """,
                    "table", {"h": 8, "w": 12, "x": 0, "y": 24}
                ),
                sql_panel(10, "Chunk Size Optimization",
                    """
                    SELECT c.chunk_size as 'Chunk Size',
                           COUNT(q.query_id) as 'Queries',
                           ROUND(AVG(q.response_time_seconds), 3) as 'Avg Time (s)',
                           COUNT(DISTINCT c.llm_model) as 'Models Used'
                    FROM configurations c
                    LEFT JOIN queries q ON c.config_id = q.config_id
                    GROUP BY c.chunk_size
                    HAVING COUNT(q.query_id) > 0
                    ORDER BY 'Avg Time (s)' ASC
                    """,
                    "table", {"h": 8, "w": 12, "x": 12, "y": 24}
                ),
                # Advanced Comparison Charts
                sql_panel(11, "Top-K vs Performance",
                    """
                    SELECT c.top_k as 'Top K',
                           COUNT(q.query_id) as 'Queries',
                           ROUND(AVG(q.response_time_seconds), 3) as 'Avg Time (s)',
                           ROUND(MIN(q.response_time_seconds), 3) as 'Min Time (s)',
                           ROUND(MAX(q.response_time_seconds), 3) as 'Max Time (s)'
                    FROM configurations c
                    LEFT JOIN queries q ON c.config_id = q.config_id
                    GROUP BY c.top_k
                    HAVING COUNT(q.query_id) > 0
                    ORDER BY c.top_k
                    """,
                    "table", {"h": 8, "w": 12, "x": 0, "y": 32}
                ),
                sql_panel(12, "Enhanced Processing Impact",
                    """
                    SELECT
                        CASE WHEN c.enhanced_processing THEN 'Enhanced' ELSE 'Basic' END as 'Processing Mode',
                        COUNT(q.query_id) as 'Queries',
                        ROUND(AVG(q.response_time_seconds), 3) as 'Avg Time (s)',
                        ROUND(MIN(q.response_time_seconds), 3) as 'Min Time (s)',
                        ROUND(MAX(q.response_time_seconds), 3) as 'Max Time (s)'
                    FROM configurations c
                    LEFT JOIN queries q ON c.config_id = q.config_id
                    GROUP BY c.enhanced_processing
                    HAVING COUNT(q.query_id) > 0
                    ORDER BY 'Avg Time (s)' ASC
                    """,
                    "table", {"h": 8, "w": 12, "x": 12, "y": 32}
                ),
                # Time-based Performance Analysis
                sql_panel(13, "Hourly Performance Patterns",
                    """
                    SELECT
                        strftime('%H', q.asked_at) as 'Hour',
                        COUNT(q.query_id) as 'Queries',
                        ROUND(AVG(q.response_time_seconds), 3) as 'Avg Time (s)'
                    FROM queries q
                    WHERE q.asked_at >= datetime('now', '-7 days')
                    GROUP BY strftime('%H', q.asked_at)
                    ORDER BY 'Hour'
                    """,
                    "table", {"h": 8, "w": 12, "x": 0, "y": 40}
                ),
                sql_panel(14, "Best Configuration",
                    """
                    SELECT
                        c.config_id as 'Config ID',
                        c.llm_provider as 'LLM Provider',
                        c.llm_model as 'Model',
                        c.embedding_model as 'Embedding',
                        c.vector_db as 'Vector DB',
                        c.chunk_size as 'Chunk Size',
                        c.top_k as 'Top K',
                        COUNT(q.query_id) as 'Queries',
                        ROUND(AVG(q.response_time_seconds), 3) as 'Avg Time (s)'
                    FROM configurations c
                    LEFT JOIN queries q ON c.config_id = q.config_id
                    WHERE q.response_time_seconds IS NOT NULL
                    GROUP BY c.config_id
                    HAVING COUNT(q.query_id) >= 3
                    ORDER BY AVG(q.response_time_seconds) ASC
                    LIMIT 1
                    """,
                    "table", {"h": 8, "w": 12, "x": 12, "y": 40}
                )
            ]
        },
        "overwrite": True
    }

    try:
        print(f"🔍 Sending dashboard config (first 500 chars): {json.dumps(dashboard_config, indent=2)[:500]}...")
        print(f"🔍 Total panels being created: {len(dashboard_config['dashboard']['panels'])}")

        response = requests.post(
            f"{GRAFANA_URL}/api/dashboards/db",
            json=dashboard_config,
            auth=(GRAFANA_USER, GRAFANA_PASS),
            headers={"Content-Type": "application/json"}
        )

        print(f"🔍 Dashboard API Response:")
        print(f"   Status Code: {response.status_code}")
        print(f"   Headers: {dict(response.headers)}")
        print(f"   Response Body: {response.text}")

        if response.status_code == 200:
            response_data = response.json()
            dashboard_url = response_data.get("url", "")
            print(f"🔍 Dashboard ID: {response_data.get('id', 'Unknown')}")
            print(f"🔍 Dashboard UID: {response_data.get('uid', 'Unknown')}")
            print(f"🔍 Dashboard Version: {response_data.get('version', 'Unknown')}")
            print(f"✅ Dashboard created successfully!")
            print(f"🔗 Dashboard URL: {GRAFANA_URL}{dashboard_url}")
            return True
        else:
            print(f"❌ Failed to create dashboard: {response.status_code}")
            try:
                error_data = response.json()
                print(f"🔍 Dashboard Error details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"🔍 Raw dashboard error response: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Error creating dashboard: {e}")
        return False


def setup_alerts():
    print("⚠️ Alert setup requires manual configuration in Grafana UI")