FROM grafana/grafana:latest

# Install SQLite datasource plugin
RUN grafana-cli plugins install frser-sqlite-datasource

# Set proper permissions
USER root
# RUN chown -R grafana:grafana /var/lib/grafana
USER grafana

# Expose port
EXPOSE 3000