# Use official Milvus image as base
FROM milvusdb/milvus:v2.6.2

# Set working directory
WORKDIR /milvus/configs

# Copy configuration files into the container
COPY embedEtcd.yaml /milvus/configs/embedEtcd.yaml
COPY user.yaml /milvus/configs/user.yaml

# Expose ports
EXPOSE 19530 9091 2379

# Health check
HEALTHCHECK --interval=30s --timeout=20s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:9091/healthz || exit 1

# Run Milvus standalone with embedded etcd
CMD ["milvus", "run", "standalone"]
