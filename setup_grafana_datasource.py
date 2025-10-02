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

    # Get absolute path to database
    db_path = os.path.abspath("rag_logs.db")

    # Check if database exists
    if not os.path.exists(db_path):
        print(f"❌ Database not found at {db_path}")
        print("💡 Make sure to run your RAG system first to create the database!")
        return False

    datasource_config = {
        "name": "RAG_SQLite",
        "type": "frser-sqlite-datasource",
        "url": "",
        "access": "proxy",
        "isDefault": True,
        "jsonData": {
            "path": db_path
        }
    }

    try:
        response = requests.post(
            f"{GRAFANA_URL}/api/datasources",
            json=datasource_config,
            auth=(GRAFANA_USER, GRAFANA_PASS),
            headers={"Content-Type": "application/json"}
        )

        if response.status_code in [200, 409]:  # 409 = already exists
            print("✅ SQLite datasource created successfully!")
            return True
        else:
            print(f"❌ Failed to create datasource: {response.status_code}")
            print(response.text)
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to Grafana: {e}")
        return False

def create_dashboard():
    """Create RAG performance dashboard"""
    print("📈 Creating RAG Performance Dashboard...")

    dashboard_config = {
        "dashboard": {
            "id": None,
            "title": "RAG Performance Monitor",
            "tags": ["rag", "performance"],
            "timezone": "browser",
            "refresh": "30s",
            "time": {
                "from": "now-24h",
                "to": "now"
            },
            "panels": [
                {
                    "id": 1,
                    "title": "Total Queries",
                    "type": "stat",
                    "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
                    "targets": [
                        {
                            "expr": "SELECT COUNT(*) as value FROM queries",
                            "refId": "A",
                            "datasource": "RAG_SQLite"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "thresholds"},
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": 0},
                                    {"color": "yellow", "value": 100},
                                    {"color": "red", "value": 1000}
                                ]
                            }
                        }
                    }
                },
                {
                    "id": 2,
                    "title": "Average Response Time",
                    "type": "stat",
                    "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0},
                    "targets": [
                        {
                            "expr": "SELECT AVG(response_time_seconds) as value FROM queries WHERE response_time_seconds IS NOT NULL",
                            "refId": "A",
                            "datasource": "RAG_SQLite"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "s",
                            "color": {"mode": "thresholds"},
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": 0},
                                    {"color": "yellow", "value": 2},
                                    {"color": "red", "value": 5}
                                ]
                            }
                        }
                    }
                },
                {
                    "id": 3,
                    "title": "Average User Rating",
                    "type": "stat",
                    "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0},
                    "targets": [
                        {
                            "expr": """
                            SELECT AVG(COALESCE(uf.rating, 0)) as value
                            FROM user_feedback uf
                            JOIN responses r ON uf.response_id = r.response_id
                            """,
                            "refId": "A",
                            "datasource": "RAG_SQLite"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "max": 5,
                            "min": 0,
                            "color": {"mode": "thresholds"},
                            "thresholds": {
                                "steps": [
                                    {"color": "red", "value": 0},
                                    {"color": "yellow", "value": 3},
                                    {"color": "green", "value": 4}
                                ]
                            }
                        }
                    }
                },
                {
                    "id": 4,
                    "title": "Active Configurations",
                    "type": "stat",
                    "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0},
                    "targets": [
                        {
                            "expr": "SELECT COUNT(DISTINCT config_id) as value FROM queries WHERE asked_at >= datetime('now', '-24 hours')",
                            "refId": "A",
                            "datasource": "RAG_SQLite"
                        }
                    ]
                },
                {
                    "id": 5,
                    "title": "Response Time Over Time",
                    "type": "timeseries",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                    "targets": [
                        {
                            "expr": """
                            SELECT
                                datetime(asked_at) as time,
                                AVG(response_time_seconds) as value
                            FROM queries
                            WHERE response_time_seconds IS NOT NULL
                            AND asked_at >= datetime('now', '-24 hours')
                            GROUP BY datetime(substr(asked_at, 1, 16) || ':00')
                            ORDER BY time
                            """,
                            "refId": "A",
                            "datasource": "RAG_SQLite"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "s",
                            "color": {"mode": "palette-classic"}
                        }
                    }
                },
                {
                    "id": 6,
                    "title": "Queries Per Hour",
                    "type": "timeseries",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                    "targets": [
                        {
                            "expr": """
                            SELECT
                                datetime(substr(asked_at, 1, 14) || '00:00') as time,
                                COUNT(*) as value
                            FROM queries
                            WHERE asked_at >= datetime('now', '-24 hours')
                            GROUP BY datetime(substr(asked_at, 1, 14) || '00:00')
                            ORDER BY time
                            """,
                            "refId": "A",
                            "datasource": "RAG_SQLite"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "palette-classic"}
                        }
                    }
                },
                {
                    "id": 7,
                    "title": "LLM Provider Performance",
                    "type": "table",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
                    "targets": [
                        {
                            "expr": """
                            SELECT
                                c.llm_provider as "Provider",
                                c.llm_model as "Model",
                                COUNT(q.query_id) as "Queries",
                                ROUND(AVG(q.response_time_seconds), 3) as "Avg Time (s)",
                                ROUND(AVG(COALESCE(uf.rating, 0)), 2) as "Avg Rating"
                            FROM configurations c
                            LEFT JOIN queries q ON c.config_id = q.config_id
                            LEFT JOIN responses r ON q.query_id = r.query_id
                            LEFT JOIN user_feedback uf ON r.response_id = uf.response_id
                            GROUP BY c.llm_provider, c.llm_model
                            HAVING COUNT(q.query_id) > 0
                            ORDER BY "Avg Rating" DESC
                            """,
                            "refId": "A",
                            "datasource": "RAG_SQLite"
                        }
                    ]
                },
                {
                    "id": 8,
                    "title": "Configuration Performance",
                    "type": "table",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
                    "targets": [
                        {
                            "expr": """
                            SELECT
                                c.chunk_size as "Chunk Size",
                                c.top_k as "Top K",
                                CASE WHEN c.enhanced_processing THEN 'Enhanced' ELSE 'Basic' END as "Mode",
                                COUNT(q.query_id) as "Queries",
                                ROUND(AVG(q.response_time_seconds), 3) as "Avg Time (s)",
                                ROUND(AVG(COALESCE(uf.rating, 0)), 2) as "Avg Rating"
                            FROM configurations c
                            LEFT JOIN queries q ON c.config_id = q.config_id
                            LEFT JOIN responses r ON q.query_id = r.query_id
                            LEFT JOIN user_feedback uf ON r.response_id = uf.response_id
                            GROUP BY c.config_id
                            HAVING COUNT(q.query_id) > 0
                            ORDER BY "Avg Rating" DESC
                            LIMIT 10
                            """,
                            "refId": "A",
                            "datasource": "RAG_SQLite"
                        }
                    ]
                }
            ]
        },
        "overwrite": True
    }

    try:
        response = requests.post(
            f"{GRAFANA_URL}/api/dashboards/db",
            json=dashboard_config,
            auth=(GRAFANA_USER, GRAFANA_PASS),
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            dashboard_url = response.json().get("url", "")
            print(f"✅ Dashboard created successfully!")
            print(f"🔗 Dashboard URL: {GRAFANA_URL}{dashboard_url}")
            return True
        else:
            print(f"❌ Failed to create dashboard: {response.status_code}")
            print(response.text)
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Error creating dashboard: {e}")
        return False

def setup_alerts():
    """Set up basic alerting rules"""
    print("🔔 Setting up alert rules...")

    # Alert for high response times
    alert_rule = {
        "alert": {
            "name": "High RAG Response Time",
            "message": "RAG system response time is above 5 seconds",
            "frequency": "1m",
            "conditions": [
                {
                    "query": {
                        "queryType": "",
                        "refId": "A",
                        "datasourceUid": "RAG_SQLite",
                        "model": {
                            "expr": "SELECT AVG(response_time_seconds) FROM queries WHERE asked_at >= datetime('now', '-5 minutes')"
                        }
                    },
                    "reducer": {
                        "type": "last",
                        "params": []
                    },
                    "evaluator": {
                        "params": [5],
                        "type": "gt"
                    }
                }
            ],
            "executionErrorState": "alerting",
            "noDataState": "no_data",
            "for": "2m"
        }
    }

    print("⚠️  Alert setup requires manual configuration in Grafana UI")
    print("💡 Go to Alerting > Alert Rules to set up notifications")

def main():
    """Main setup function"""
    print("🚀 Setting up Grafana for RAG Performance Monitoring")
    print("=" * 50)

    # Wait for Grafana to be ready
    if not wait_for_grafana():
        print("❌ Grafana is not responding. Please check if it's running.")
        print("💡 Run: sudo systemctl start grafana-server")
        return False

    # Create datasource
    if not create_datasource():
        return False

    # Small delay to ensure datasource is ready
    time.sleep(2)

    # Create dashboard
    if not create_dashboard():
        return False

    # Setup alerts info
    setup_alerts()

    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print(f"📊 Open your dashboard: {GRAFANA_URL}")
    print("🔑 Login: admin / admin (change password on first login)")
    print("\n💡 Tips:")
    print("- Dashboard auto-refreshes every 30 seconds")
    print("- Click on panels to drill down into data")
    print("- Use time range picker to analyze different periods")
    print("- Set up email/Slack notifications in Alerting section")

    return True

if __name__ == "__main__":
    main()