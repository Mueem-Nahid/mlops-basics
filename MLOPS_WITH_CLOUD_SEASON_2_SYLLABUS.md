# MLOPS WITH CLOUD SEASON 2 - Course Syllabus

**Course URL:** https://poridhi.io/course-details/689c8df7a1f6c3e5abdfefba

---

## Prerequisites

### ML Prerequisites
- Basic Data Transformation
- Basic Understanding of Different Model Architectures in different domains (CV, NLP)
- Model training with different architectures (Classical ML, NN, CNN, RNN etc.)

### Other Prerequisites
- Basic understanding of Linux, WSL2 and Git/GitHub

---

## Module 1: Experiment Tracking & Model Metadata Management (MLflow)

- MLflow components: tracking, registry, models, projects
- Deploy MLflow on AWS EC2 with RDS (Postgres) and S3 artifact storage
- Authentication and TLS setup with Nginx reverse proxy
- Logging metrics, parameters, and artifacts
- Integrating MLflow with scikit-learn, PyTorch, TensorFlow

---

## Module 2: Data & Pipeline Versioning (DVC, git-lfs)

- Versioning datasets alongside code
- Using DVC with S3 remote storage
- Git LFS for large file handling
- Data lineage and reproducibility with DVC
- Integrating DVC with MLflow

---

## Module 3: Feature Stores & Databases (Feast, Redis, Postgres)

- Feature store fundamentals: entities, feature views, materialization
- Deploy Feast with S3 offline store and Redis online store on AWS
- Freshness metrics and monitoring feature health
- Online/offline feature consistency

---

## Module 4: API Development with FastAPI for Model Serving

- Designing inference APIs for ML models
- Loading and serving MLflow-registered models in FastAPI
- Securing endpoints with API keys and rate limiting
- Instrumentation with Prometheus metrics endpoints

---

## Module 5: CI/CD for ML Services

- GitHub Actions workflows for ML model CI
- Docker build & push to AWS ECR
- Canary and blue/green deployment strategies overview
- Connecting CI/CD with MLflow model promotion

---

## Module 6: Monitoring & Observability for ML Systems

- Metrics: infra, application, model
- Deploy Prometheus & Grafana on AWS EC2
- Dashboarding model accuracy, latency, drift indicators
- Alerting with Alertmanager + SNS

---

## Module 7: AWS Fundamentals for MLOps

- AWS global infrastructure overview
- Regions, AZs, VPC basics
- IAM essentials (users, roles, policies, least privilege)
- Cost guardrails: Budgets, billing alerts, tagging strategies

---

## Module 8: Compute & Storage for ML

- EC2 instance types for ML workloads
- Configuring S3 buckets for ML data and artifacts
- S3 lifecycle policies & storage classes
- Using RDS (Postgres/MySQL) for ML metadata

---

## Module 9: Networking for ML Systems

- VPC design for ML workloads
- Subnets (public vs private), route tables, NAT gateways
- Security groups and NACLs
- PrivateLink and VPC endpoints for secure AWS service access

---

## Module 10: Load Balancing, Scaling & Reliability

- Elastic Load Balancing (ALB/NLB) for ML APIs
- Auto Scaling Groups for serving workloads
- Scaling policies for latency, CPU, or custom metrics
- High availability patterns in AWS

---

## Module 11: Infrastructure as Code for AWS

- Terraform & Pulumi basics for AWS infra
- Modular infrastructure patterns
- Using variables, workspaces, and state management
- CI/CD for IaC

---

## Module 12: Security, Compliance & Cost Optimization

- Encrypting data in transit & at rest (KMS, ACM)
- WAF & Shield for API security
- GuardDuty for threat detection
- Cost optimization strategies for EC2, S3, RDS
- Instance right-sizing & Spot Instances

---

## Module 13: Docker Fundamentals for ML Workloads

- What is containerization and why it matters in MLOps
- Comparing VMs and containers (pros, cons, resource usage)
- Installing and configuring Docker for development
- Understanding images, containers, layers, and registries
- The role of containerization in reproducible ML experiments

---

## Module 14: Writing Dockerfiles for ML Applications

- Best practices for structuring Dockerfiles for Python/ML projects
- Multi-stage builds to reduce image size
- Managing Python dependencies with pip, poetry, or conda in containers
- Using .dockerignore to reduce build context size
- Incorporating system-level dependencies (OpenCV, CUDA libraries, etc.)

---

## Module 15: Containerizing ML APIs and Batch Jobs

- Packaging a FastAPI model-serving service into a container
- Building containers for data preprocessing and ETL jobs
- Entrypoints and CMD for batch processing containers
- Passing environment variables and secrets securely
- Performance considerations (CPU pinning, memory limits, caching layers)

---

## Module 16: Managing Docker Images & Registries

- Using Docker Hub, AWS ECR
- Versioning ML service images for rollback and reproducibility
- Image scanning for vulnerabilities (Trivy, Grype)
- Automating builds and pushes via CI/CD pipelines
- Cleaning up unused images and layers to reduce costs

---

## Module 17: Multi-Container Architectures for ML Systems

- Using Docker Compose for local multi-service ML stacks (API + DB + monitoring)
- Defining service dependencies (e.g., FastAPI + Redis + MLflow + Kafka)
- Networking containers together
- Sharing volumes for feature stores and artifact storage
- Local dev workflow for end-to-end pipelines

---

## Module 18: GPU-Accelerated Containers for ML/DL

- Installing NVIDIA Container Toolkit for GPU support
- Building CUDA-enabled images for PyTorch/TensorFlow
- GPU resource isolation in Docker
- Testing and benchmarking GPU containers locally before cloud deployment

---

## Module 19: Debugging & Optimizing Containers

- Inspecting running containers (logs, exec, stats)
- Measuring container resource usage (CPU, memory, GPU, network)
- Reducing cold-start latency for ML APIs
- Caching dependencies for faster builds
- Handling container crashes and restart policies

---

## Module 20: Data Engineering Concepts for MLOps

- The role of data engineering in production ML systems
- Batch vs streaming pipelines in ML use cases
- Latency, throughput, and freshness trade-offs
- Data schema evolution and governance for ML
- Common pitfalls (e.g., train–serve skew, stale features, poor data quality)

---

## Module 21: Event Streaming with Kafka

- Kafka architecture for ML: brokers, topics, partitions, replication
- Kafka KRaft vs ZooKeeper mode
- Designing topics for ML use cases (keying, partitioning, retention)
- Installing Kafka (local + AWS EC2)
- Producers & consumers in Python (confluent-kafka, aiokafka)
- Kafka Connect for ingestion (S3 sink, JDBC sink)
- Schema Registry (Avro/Protobuf/JSON) for ML feature contracts
- Metrics & monitoring (lag, ISR, partition skew)

---

## Module 22: Batch Processing with Apache Spark

- Spark fundamentals (RDD vs DataFrame API)
- Reading from S3, JDBC, and Kafka
- Transformations for ML (feature engineering in Spark)
- Writing partitioned Parquet to S3 for offline ML training
- Spark on AWS EMR vs standalone cluster on EC2
- Optimization techniques (partitioning, bucketing, caching)

---

## Module 23: Streaming Feature Pipelines

- Use cases for streaming in ML (fraud detection, recommender freshness, real-time personalization)
- Building sliding window aggregations with Kafka Streams & Faust
- Joining real-time data with static reference datasets
- Handling late/out-of-order data
- Monitoring freshness & processing latency

---

## Module 24: Data Quality for ML

- Why ML needs stricter data quality checks
- Great Expectations for batch pipelines
- Pandera for Python dataframe validation
- Detecting drift at the data layer (statistical checks pre-model)
- Automated quality gates in CI/CD pipelines

---

## Module 25: Workflow Orchestration with Airflow

- Airflow basics: DAGs, tasks, scheduling, retries
- Setting up Airflow locally & on AWS EC2
- Integrating batch feature engineering with Airflow DAGs
- Sensors for data availability checks
- Airflow with KubernetesPodOperator for scalable ML jobs

---

## Module 26: Integration with Feature Stores

- Role of feature stores in MLOps
- Connecting Kafka/Spark outputs to Feast
- Offline vs online stores (S3/Parquet vs Redis/Postgres)
- TTL & freshness guarantees
- Example: ingesting Spark output to Feast for model training & real-time lookup

---

## Module 27: Project Scaffolding, Environment Setup & Cost Guardrails

- Structuring the repository for an MLOps project (infra/, services/, features/, mlops/)
- Setting up pre-commit hooks, linting, and tests (ruff, black, pytest)
- AWS CLI profiles, SSM Parameter Store for secrets
- AWS Budgets and cost alerts for student environments
- Writing teardown scripts for resource cleanup

---

## Module 28: Streaming Infrastructure with Kafka on AWS EC2

- Deploying a 3-broker Kafka cluster on EC2 with schema registry
- Setting replication factor and ISR settings for fault tolerance
- Creating topics for transactions, scores, and dead letters
- Adding Kafka exporters for monitoring
- Schema evolution and compatibility tests

---

## Module 29: Raw Event Storage (S3 + MongoDB)

- Streaming data ingestion from Kafka to S3 in partitioned Parquet format
- Setting up MongoDB replica set for low-latency recent lookups
- Configuring Kafka Connect S3 and Mongo sinks
- TTL indexes in Mongo for cost control
- Verifying data freshness and schema compliance

---

## Module 30: Experiment Tracking & Model Registry with MLflow

- Deploying MLflow on EC2 with RDS (PostgreSQL) backend and S3 artifact store
- Enabling TLS and authentication via Nginx reverse proxy
- Logging metrics, parameters, and artifacts from experiments
- Registering and versioning models in the MLflow Model Registry
- Integrating Prometheus metrics from MLflow

---

## Module 31: Baseline Model Training & Logging

- Loading IEEE-CIS fraud dataset from S3
- Performing time-based train-test splits to avoid leakage
- Handling class imbalance (class weights, resampling)
- Evaluating with PR-AUC, ROC-AUC, calibration curves
- Logging all runs to MLflow with reproducibility artifacts

---

## Module 32: Feature Store with Feast

- Defining entities, features, and TTL policies
- Using S3 for offline storage, Redis for online storage
- Materializing features and performing online lookups
- Integrating feature freshness and hit/miss metrics into Prometheus

---

## Module 33: Real-Time Feature Aggregation

- Implementing streaming aggregations (e.g., 5-min, 30-min transaction counts) with Kafka Streams or Faust
- Backfilling features for historical data
- Ensuring idempotency and correctness in streaming updates

---

## Module 34: Model Serving — Streaming Scoring Service

- Building a Kafka consumer service to:
  - Fetch features from Feast
  - Score using MLflow model
  - Publish scores to Kafka and S3
  - Handle failed events with a dead-letter topic
- Performance tuning for p95 latency under 150ms

---

## Module 35: Ad-hoc Prediction API with FastAPI

- Implementing an API for investigation teams
- Single and batch prediction endpoints
- API key authentication and rate limiting
- Exposing Prometheus metrics for API health

---

## Module 36: Containerization & Deployment to AWS

- Containerizing all services with Docker
- Multi-arch builds (ARM/x86) for EC2 Graviton
- Pushing to AWS ECR with automated scans
- Deploying scoring service and API in an Auto Scaling Group with ALB

---

## Module 37: CI/CD for Fraud Detection Services

- Building GitHub Actions workflows for building, testing, and deploying services
- Canary and blue/green deployments with AWS CodeDeploy

---

## Module 38: Monitoring & Observability

- Setting up Prometheus on EC2 to scrape exporters from all components
- Grafana dashboards for:
  - Inference latency
  - Kafka consumer lag
  - Feature freshness
  - Model health metrics
- Alerting on SLO violations with Alertmanager + SNS/Slack

---

## Module 39: Load Testing & Latency Optimization

- Running load tests with Locust or k6
- Identifying bottlenecks in feature lookup, model scoring, or Kafka
- Tuning workers, connection pools, and caching layers

---

## Module 40: Continual Learning & Drift Detection

- Implementing Evidently AI for concept drift detection
- Triggering retraining pipelines on drift events
- Automating model evaluation gates before promotion

---

## Module 41: Kubernetes Fundamentals for MLOps

- Why Kubernetes is essential for modern ML pipelines
- Kubernetes architecture: API server, scheduler, controller manager, kubelet, etcd
- Pods, ReplicaSets, Deployments, Services, and Ingress
- Kubernetes vs Docker Compose for ML workloads
- Local development with K3s

---

## Module 42: Setting Up Kubernetes Environments (K3s & AWS EKS)

- Installing K3s locally with Helm support
- Creating an EKS cluster with Terraform (IaC approach)
- Understanding EKS networking (VPC, CNI, security groups, IAM roles)
- kubectl configuration & context switching between clusters

---

## Module 43: Kubernetes Resource Management for ML

- CPU, GPU, and memory requests & limits
- GPU scheduling in Kubernetes (NVIDIA device plugin)
- Node affinity, taints, and tolerations for ML workloads
- Autoscaling: HPA (Horizontal Pod Autoscaler) & VPA (Vertical Pod Autoscaler)

---

## Module 44: Storage & Data in Kubernetes

- Persistent Volumes (PV) and Persistent Volume Claims (PVC) for ML
- Mounting S3 buckets to pods (S3 CSI driver)
- Connecting PVCs to ML training jobs
- Managing feature store data in Feast with Kubernetes storage

---

## Module 45: Networking & Service Exposure

- ClusterIP, NodePort, LoadBalancer, and Ingress for ML services
- NGINX ingress controller setup
- Securing ML endpoints with TLS and authentication
- Internal vs public-facing ML APIs

---

## Module 46: Configurations, Secrets, and Environment Management

- ConfigMaps for environment-specific configs
- Secrets management with Kubernetes Secrets and AWS Secrets Manager
- Rolling updates with zero downtime for ML models
- Versioning model configurations

---

## Module 47: Observability for ML Workloads in Kubernetes

- Integrating Prometheus & Grafana with Kubernetes metrics
- Collecting ML-specific metrics from pods (latency, throughput, drift)
- Logging with Loki or EFK stack

---

## Module 48: CI/CD for Kubernetes MLOps

- GitHub Actions → EKS deploy pipeline
- Canary and Blue/Green deployments for ML models
- Rollbacks on performance degradation

---

## Module 49: Introduction to Kubeflow for MLOps

- What is Kubeflow? Why it matters for production ML
- Core components: Pipelines, KFServing, Katib, Metadata
- Kubeflow vs Airflow vs Argo Workflows
- Real-world use cases of Kubeflow in ML teams
- Lab: Install Kubeflow locally (MiniKF) and explore the central dashboard

---

## Module 50: Setting Up Kubeflow on Kubernetes

- Deploying Kubeflow on AWS EKS using manifests/Helm
- Configuring authentication (Dex, OIDC)
- Integrating with AWS S3 for artifact storage
- Connecting Kubeflow to an external MLflow tracking server
- Lab: Deploy Kubeflow on AWS EKS with S3 as artifact store

---

## Module 51: Creating Your First ML Pipeline

- Creating ML Pipeline with Kubeflow
- Writing Kubeflow components in Python
- Passing data and artifacts between pipeline steps
- Versioning pipelines and tracking executions
- Lab: Create a pipeline that loads data, trains a model, and logs metrics to MLflow

---

## Module 52: Advanced Pipeline Patterns

- Parallelism and conditional execution
- Caching and skipping steps for faster reruns
- Reusable components and shared libraries
- Multi-model pipelines (ensembles, experiments)
- Lab: Build a pipeline that trains multiple models in parallel and selects the best one

---

## Module 53: Hyperparameter Tuning with Katib

- Katib architecture and integration with Kubeflow Pipelines
- Defining search spaces and objectives
- Distributed hyperparameter tuning
- Logging Katib results to MLflow
- Lab: Run Katib experiments for model tuning on EKS

---

## Module 54: Continuous Training & Deployment Pipelines

- Automating retraining with new data triggers
- CI/CD for pipelines using GitHub Actions and ArgoCD
- Canary and blue/green deployments with KFServing
- Integrating model registry (MLflow/Kubeflow)
- Lab: Build a continuous training pipeline with KFServing model deployment

---

## Module 55: Serving Models with KFServing

- KFServing architecture and model serving patterns
- Deploying models as REST and gRPC endpoints
- Scaling inference services with autoscaling
- A/B testing models in production
- Lab: Deploy a trained model using KFServing with canary rollout

---

## Module 56: Monitoring & Observability in Kubeflow

- Pipeline run metadata tracking
- Exporting pipeline metrics to Prometheus/Grafana
- Model performance monitoring in production
- Drift detection integration with Evidently
- Lab: Build Grafana dashboards for Kubeflow Pipelines and model health

---

## Module 57: Best Practices & Cost Optimization

- Designing modular pipelines for reusability
- Securing Kubeflow in multi-tenant environments
- Reducing cloud costs with spot instances and caching
- Backup and disaster recovery for pipeline metadata
- Lab: Implement cost-optimized pipeline execution with spot nodes and caching

---

## Module 58: Problem Definition & System Architecture

- Understanding recommendation problem types (content-based, collaborative filtering, hybrid)
- Business and ML requirements for recommender systems
- High-level architecture on Kubernetes (data → features → model → serving → monitoring)
- Components: Kubeflow Pipelines, Ray, Feast, KFServing, Prometheus/Grafana

---

## Module 59: Data Ingestion & Processing

- Sources: user interactions, item metadata, transaction history
- Batch vs streaming ingestion in Kubernetes
- Ingestion pipeline with Kafka for streaming events and Spark for batch
- Schema design for recommendation datasets

---

## Module 60: Feature Engineering with Feast

- Setting up Feast in Kubernetes (Redis + S3)
- Defining entities, features, and feature views for recommendations
- Materializing offline → online store
- Handling TTL and freshness in real-time features

---

## Module 61: Distributed Training with Ray on Kubernetes

- Ray cluster setup on Kubernetes
- Parallelizing model training (matrix factorization, deep learning-based recommenders)
- Ray Tune for hyperparameter optimization
- Integrating Ray with MLflow for experiment tracking

---

## Module 62: Orchestrating the Pipeline with Kubeflow

- Writing Kubeflow components for data ingestion, feature generation, model training, evaluation, and deployment
- Defining pipeline parameters (dataset, hyperparameters, model type)
- Scheduling periodic runs

---

## Module 63: Model Evaluation & A/B Testing

- Offline evaluation metrics: precision@k, recall@k, MAP, NDCG
- Online evaluation: A/B tests with live traffic
- Canary deployments and traffic splitting with KFServing

---

## Module 64: Real-Time Serving with KFServing

- Deploying recommendation models as scalable REST endpoints
- Handling feature lookups in real-time requests
- Scaling inference with Ray Serve

---

## Module 65: Monitoring & Observability

- Tracking latency, throughput, and error rates for inference services
- Monitoring model performance drift in recommendation quality
- Creating Grafana dashboards for user engagement metrics

---

## Module 66: Cost Optimization & Scaling

- Scaling Ray workers on spot instances
- Optimizing Kubernetes autoscaling for batch and inference workloads
- Reducing pipeline execution costs

---

## Module 67: Deep Learning Fundamentals in the MLOps Context

- Why deep learning in production differs from academic DL
- Brief overview of CNNs, RNNs, and Transformer-based architectures
- Understanding compute requirements: CPU vs GPU vs TPU
- Batch vs online inference
- Data dependencies and versioning for DL workloads
- Reproducibility in DL pipelines (seed setting, deterministic ops, containerized environments)

---

## Module 68: Data Preprocessing Pipelines for DL

- Scalable preprocessing with Spark or Ray Data
- Augmentation strategies for CV, NLP, and audio tasks
- Ensuring consistent preprocessing in training & serving (feature parity)
- Caching preprocessed datasets for speed and cost optimization
- Using tf.data pipelines or PyTorch DataLoader for efficient streaming

---

## Module 69: Training Deep Learning Models at Scale

- Multi-GPU training (Data Parallelism, Model Parallelism)
- Mixed precision training for performance gains
- Distributed training with Ray Train or PyTorch DDP
- Hyperparameter tuning for DL (Ray Tune, Optuna)
- Logging metrics, losses, and model artifacts in MLflow

---

## Module 70: Model Packaging & Versioning

- Exporting models in multiple formats
- Model signatures and schema validation
- Storing and managing versions in MLflow Model Registry
- Automated CI tests for model compatibility before deployment

---

## Module 71: Serving Deep Learning Models

- FastAPI + Uvicorn/Gunicorn for DL inference APIs
- Batch vs real-time endpoints
- Using Ray Serve or TorchServe for scaling inference
- GPU scheduling & resource allocation
- Handling large models with lazy loading and warmup strategies
- Integrating Prometheus metrics for inference performance

---

## Module 72: GPU Inference Optimization

- TensorRT optimization
- Quantization (dynamic, post-training, quantization-aware)
- Model pruning and distillation for latency reduction
- Profiling inference performance with NVIDIA Nsight and PyTorch profiler
- Serving optimized models in production

---

## Module 73: CI/CD for Deep Learning Pipelines

- Building inference images with GPU base containers
- Testing model performance in staging before promotion
- Canary releases for DL models
- Automating redeployment when a new model version passes benchmarks

---

## Module 74: Monitoring DL Models in Production

- Latency, throughput, and GPU utilization tracking
- Triggering retraining workflows for DL models

---

## Module 75: Project Scaffolding, Environment Setup & Cost Guardrails

- Repository structure for streaming anomaly detection (infra/, services/, features/, mlops/)
- Pre-commit hooks, linting, testing (ruff, black, pytest)
- AWS CLI profiles, SSM Parameter Store for secrets
- AWS Budgets and SNS alerts for infra spend
- Automated teardown scripts for all AWS resources

---

## Module 76: Streaming Infrastructure with Kafka on AWS EC2

- Deploy 3-broker Kafka cluster with schema registry
- Configure replication factor, ISR, and retention for time-series workloads
- Topics: raw_events, anomaly_scores, alerts, dead_letters
- Enable Kafka JMX exporter for monitoring
- Schema evolution and backward compatibility testing

---

## Module 77: Raw Event Storage & Historical Store

- Stream ingestion from Kafka to S3 in partitioned Parquet format (for batch retraining)
- MongoDB or PostgreSQL for low-latency lookup of recent events
- Kafka Connect S3/Mongo/Postgres sinks
- TTL indexes for storage cost optimization
- Data freshness verification pipelines

---

## Module 78: Experiment Tracking & Model Registry with MLflow

- Deploy MLflow on EC2 with RDS + S3 backend
- Secure with TLS + Nginx reverse proxy
- Log anomaly detection experiments (Isolation Forest, Autoencoders, LSTM, etc.)
- Version models in MLflow Model Registry
- Expose metrics from MLflow to Prometheus

---

## Module 79: Baseline Model Training & Evaluation

- Use simulated IoT/financial/log data for anomalies
- Handle extreme class imbalance
- Evaluate with precision-recall curves, F1@fixed recall, anomaly score distributions
- Log parameters, metrics, and artifacts to MLflow
- Store train/test splits in DVC for reproducibility

---

## Module 80: Feature Store with Feast

- Define entities (device_id, account_id) and anomaly-relevant features
- Use S3 for offline store, Redis for online store
- Materialize real-time features for streaming scoring
- Monitor feature hit/miss ratio with Prometheus

---

## Module 81: Real-Time Feature Aggregation

- Compute rolling statistics (mean, std dev, min/max) over multiple windows (5 min, 1 hr, 24 hr)
- Implement aggregations using Kafka Streams / Faust
- Backfill missing features from historical store
- Ensure consistency between batch and streaming pipelines

---

## Module 82: Model Serving — Streaming Anomaly Detection Service

- Kafka consumer fetches features from Feast
- Scores events using deployed MLflow model
- Publishes anomaly scores to Kafka + writes to S3 for audit
- Dead-letter handling for invalid events
- Maintain p95 latency < 200ms

---

## Module 83: Alerting API with FastAPI

- Expose REST API for on-demand anomaly checks
- Endpoint for batch investigation
- API key auth + rate limiting
- Prometheus metrics for API health & anomaly counts

---

## Module 84: Containerization & Deployment

- Containerize services with Docker
- Optimize images for low cold-start latency
- Push to ECR with vulnerability scans
- Deploy on EC2 Auto Scaling Group or EKS
- Load balancing via ALB/NLB

---

## Module 85: CI/CD for Anomaly Detection

- GitHub Actions workflows for test → build → deploy
- Canary deploys with AWS CodeDeploy
- Automated rollback on regression in latency or alert volume

---

## Module 86: Monitoring & Observability

- Prometheus to scrape metrics from all services
- Grafana dashboards for monitoring
- Alertmanager rules for anomaly spikes & system failures

---

## Module 87: Load Testing & Latency Optimization

- Simulate high event throughput with Locust/k6
- Identify bottlenecks in feature lookup, scoring, and Kafka consumers
- Optimize workers, concurrency, and batch processing

---

## Module 88: Continual Learning & Drift Detection

- Use Evidently AI for detecting concept drift & data drift
- Retrain model automatically on confirmed drifts
- Validate retrained model before promotion to production

---

## Module 89: Governance, Explainability & Cost Intelligence

- Version all datasets and pipelines with DVC
- Implement feature contract tests to detect schema drift
- Use SHAP/Integrated Gradients for explainability
- Track cost per 1K anomaly checks and optimize infrastructure

---

## Module 90: Problem Definition & Use Cases

- Common CV tasks: image classification, object detection, segmentation
- Use case selection (e.g., real-time defect detection, product tagging)
- Business KPIs vs ML metrics
- High-level architecture: ingestion → preprocessing → training → deployment → monitoring

---

## Module 91: Data Acquisition & Storage

- Sources: datasets (ImageNet, COCO), customer uploads, camera streams
- Batch ingestion from S3 and streaming ingestion with Kafka
- Data storage strategy in cloud (S3 bucket partitioning, lifecycle policies)

---

## Module 92: Distributed Training with Ray

- Ray cluster setup for distributed deep learning
- Integrating PyTorch DistributedDataParallel (DDP) with Ray
- Ray Tune for hyperparameter search (learning rate, batch size, augmentations)
- Tracking experiments with MLflow

---

## Module 93: Model Evaluation

- Evaluation metrics for CV (accuracy, mAP, IoU)
- Error analysis and confusion matrix interpretation
- Logging evaluation artifacts to MLflow

---

## Module 94: Model Optimization with TensorRT

- Introduction to model compression (quantization, pruning)
- Converting PyTorch/TensorFlow models to TensorRT
- Benchmarking latency & throughput improvements

---

## Module 95: Deployment with KFServing

- Containerizing the optimized model
- KFServing deployment for scalable inference
- GPU scheduling in Kubernetes
- Canary deployments for new CV model versions

---

## Module 96: Real-Time Inference

- Streaming inference pipeline
- Handling variable versions and resolution
- Scaling inference workloads in Kubernetes

---

## Module 97: Monitoring & Drift Detection

- Monitoring inference latency, FPS, and GPU utilization
- Detecting data distribution drift in images
- Integrating EvidentlyAI with Prometheus/Grafana

---

## Module 98: Continuous Training & Automation

- Automating model retraining when drift or degradation is detected
- Updating TensorRT optimizations in retraining
- Kubeflow Pipelines for automated retraining cycles

---

## Module 99: Project Scaffolding, Environment Setup & Cost Guardrails

- Repo structure for multi-model NLP project (infra/, datasets/, models/, pipelines/, services/)
- Poetry or uv for dependency management, pre-commit hooks, linting (ruff, black), testing (pytest)
- AWS CLI profiles, SSM Parameter Store for credentials
- Cost monitoring with AWS Budgets & teardown scripts
- GPU-aware environment setup (CUDA, cuDNN, NCCL)

---

## Module 100: NLP Data Engineering & Preprocessing

- Building a text ingestion pipeline from S3 + Kafka
- Data cleaning, deduplication, tokenization (Hugging Face Tokenizers, SentencePiece)
- Generating and storing embeddings in Qdrant / PostgreSQL + pgvector
- Versioning datasets with DVC (storing raw + preprocessed versions)
- Parallel preprocessing with Ray Data

---

## Module 101: Experiment Tracking & Model Registry with MLflow

- Tracking BERT fine-tunes, embedding models, and LLM pretraining runs
- Logging training loss, eval metrics, confusion matrices, embeddings visualizations
- Registering models in MLflow Model Registry with stage transitions (dev → staging → prod)
- Integrating MLflow with Ray Tune for distributed hyperparameter search

---

## Module 102: Applied NLP Model Development

- Fine-tuning BERT / RoBERTa for classification, NER, QA
- Using LoRA / PEFT for parameter-efficient fine-tuning
- Evaluating with F1, macro/micro precision-recall, exact match (QA)
- Exporting to ONNX/TensorRT for optimized inference

---

## Module 103: LLM from Scratch — Architecture & Training

- Implementing Transformer architecture (multi-head attention, feed-forward, layer norm) in PyTorch
- Pretraining on a curated corpus (wiki + domain-specific data) using Ray Train for distributed multi-GPU training
- Mixed-precision (fp16/bf16) & gradient checkpointing for efficiency
- Evaluating perplexity, next-token prediction accuracy
- Saving checkpoints to S3 with metadata for reproducibility

---

## Module 104: Feature Store for NLP Pipelines

- Using Feast to store reusable features (text embeddings, entity frequency tables)
- Redis for online store, S3 for offline store
- Materializing features for batch and streaming NLP pipelines

---

## Module 105: Deployment Infrastructure for NLP Models

- Deploying inference endpoints with Ray Serve (multi-model routing: BERT classifier, embedding service, LLM)
- Containerizing services with GPU-enabled Docker images
- Deploying on k8s with GPU nodes & autoscaling
- Load testing inference with Locust/k6 for latency and throughput

---

## Module 106: Retrieval-Augmented Generation (RAG) Pipeline

- Vector DB setup (Qdrant, OpenSearch, pgvector) for context retrieval
- Building RAG workflow for LLM with Ray Serve pipelines
- Integrating with FastAPI API layer for user queries
- Caching retrieved contexts with Redis for hot queries

---

## Module 107: API Development & Model Serving

- REST & gRPC endpoints for:
  - Classification
  - NER
  - Embedding lookups
  - RAG queries
- API key authentication & request quotas
- Prometheus metrics for request volume, latency, model hit ratios

---

## Module 108: Monitoring & Observability

- Prometheus to scrape metrics from all services
- Grafana dashboards for:
  - Inference latency (p50, p95, p99)
  - Embedding lookup times
  - RAG retrieval + generation latency
  - Model accuracy & drift
- Alerting for SLO breaches

---

## Module 109: Continual Learning & Fine-Tuning in Production

- Automating BERT/LLM fine-tuning when new labeled data arrives
- Using RLHF or DPO for LLM alignment in production
- Versioning embeddings when retrained models are promoted
- AB tests for new model versions

---

## Module 110: Project Scaffolding, Environment Setup & Cost Guardrails

- Repo structure for time-series forecasting (infra/, data/, models/, pipelines/, services/)
- Setting up Poetry/uv, pre-commit hooks, linting (ruff, black), testing (pytest)
- AWS CLI profiles, SSM for credentials
- Budgets & teardown scripts

---

## Module 111: Time Series Data Engineering & Preprocessing

- Sources: sensors, financial ticks, server logs
- Batch ingestion from S3, streaming from Kafka
- Handling missing values, outliers, resampling
- Feature engineering: lags, rolling statistics, seasonal decomposition
- Versioning with DVC

---

## Module 112: Experiment Tracking & Model Registry with MLflow

- Tracking classical TS models (ARIMA, Prophet, ETS) and deep learning models (LSTM, Transformer, TFT)
- Logging hyperparameters, evaluation metrics (MAE, RMSE, MAPE), and forecast plots
- Registering models with stage transitions (dev → staging → prod)

---

## Module 113: Baseline & Advanced Forecasting Models

- Classical models: ARIMA, Prophet, Exponential Smoothing
- Deep learning: LSTM, GRU, Temporal Fusion Transformer (TFT)
- Distributed training with Ray for large-scale time series
- Hyperparameter tuning with Ray Tune
- Logging all experiments to MLflow

---

## Module 114: Feature Store for Time Series

- Time-windowed features: rolling mean, lag, seasonal decomposition
- Using Feast for online and offline feature serving
- Backfilling historical features for training
- Real-time feature computation for streaming forecasts

---

## Module 115: Batch Forecasting Pipeline

- Kubeflow pipeline for scheduled batch forecasting
- Feature extraction from historical data
- Model loading from MLflow registry
- Generating multi-horizon forecasts
- Storing predictions in S3 and PostgreSQL

---

## Module 116: Real-Time Forecasting Pipeline

- Streaming data ingestion from Kafka
- Real-time feature computation
- Low-latency inference with cached models
- Publishing forecasts to downstream consumers
- Monitoring forecast latency and accuracy

---

## Module 117: Model Evaluation & Selection

- Metrics: MAE, RMSE, MAPE, forecast bias
- Backtesting strategies for time series
- Comparing multiple model versions
- Automated model selection based on performance

---

## Module 118: API Development & Serving

- FastAPI endpoints for on-demand forecasts
- Batch forecasting API for historical periods
- Authentication and rate limiting
- Exposing metrics to Prometheus

---

## Module 119: Monitoring & Drift Detection

- Tracking forecast accuracy over time
- Detecting distribution shifts in input data
- Automated alerts for model degradation
- Dashboard design for forecast monitoring

---

## Module 120: Continuous Training & Deployment

- Triggered retraining on new data availability
- Automated model validation before deployment
- Blue/green deployments for forecast services
- Version control and rollback mechanisms

---

## Module 121: Advanced Time Series Techniques

- Multi-variate forecasting
- Hierarchical time series reconciliation
- Probabilistic forecasting & prediction intervals
- Handling seasonality & trend changes

---

## Module 122: Scalability & Performance Optimization

- Distributed inference for large-scale forecasting
- Caching strategies for frequently accessed forecasts
- Asynchronous processing pipelines
- Cost optimization for compute resources

---

## Module 123: Production Best Practices for Time Series

- Handling data quality issues in production
- Managing forecast horizon vs accuracy tradeoffs
- Documenting model assumptions & limitations
- Creating reproducible forecasting environments

---

## Module 124: Edge Cases & Error Handling

- Handling missing data in production streams
- Dealing with anomalous input values
- Graceful degradation strategies
- Alert fatigue management

---

## Module 125: Integration & Orchestration

- Integrating forecasts with business systems
- Orchestrating multi-model ensemble forecasts
- Managing dependencies between forecasting services
- Event-driven architecture patterns

---

## Module 126: Security & Compliance

- Securing forecast APIs and data access
- Data retention policies for time series
- Audit logging for model predictions
- Compliance with data privacy regulations

---

## Module 127: Advanced MLOps Patterns

- Multi-tenancy in ML platforms
- Cross-region deployment strategies
- Disaster recovery for ML systems
- Managing technical debt in ML pipelines

---

## Module 128: Course Wrap-up & Future Directions

- Recap of key MLOps concepts covered
- Industry trends & emerging technologies
- Building a career in MLOps
- Resources for continued learning
- Final project showcase & best practices

---

## Course Summary

**Total Modules:** 128

**Key Projects:**
1. Fraud Detection System (Modules 27-40)
2. Recommender System (Modules 58-66)
3. Anomaly Detection System (Modules 75-89)
4. Computer Vision Pipeline (Modules 90-98)
5. NLP/LLM System (Modules 99-109)
6. Time Series Forecasting (Modules 110-120)

**Technologies Covered:**
- **Experiment Tracking & Versioning:** MLflow, DVC
- **Feature Stores:** Feast, Redis, PostgreSQL
- **Data Engineering:** Kafka, Spark, Airflow
- **Containerization:** Docker, Kubernetes, Helm
- **Orchestration:** Kubeflow Pipelines
- **Distributed Computing:** Ray, PyTorch DDP
- **Cloud Platform:** AWS (EC2, EKS, S3, RDS, ECR, etc.)
- **Monitoring:** Prometheus, Grafana, Evidently AI
- **API Development:** FastAPI
- **CI/CD:** GitHub Actions, AWS CodeDeploy, ArgoCD
- **Deep Learning:** PyTorch, TensorFlow, TensorRT
- **NLP/LLM:** Transformers, BERT, Ray Serve
- **Infrastructure as Code:** Terraform

**Course Focus:**
- Production-ready ML systems on cloud infrastructure
- End-to-end MLOps lifecycle
- Real-world hands-on projects across multiple domains (Fraud Detection, CV, NLP, Time Series, Recommender Systems, Anomaly Detection)
- Scalable and cost-optimized ML architectures
- Industry best practices for ML in production
