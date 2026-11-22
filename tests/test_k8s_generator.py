# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for K8sManifestGenerator, UniqueKeySafeLoader, and structural validation."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest
from devops_ai.core.exceptions import DevOpsAIError
from devops_ai.generators.k8s_manifest import K8sManifestGenerator
from devops_ai.llm.base import BaseLLMProvider


@pytest.fixture
def mock_provider():
    provider = MagicMock(spec=BaseLLMProvider)
    provider.provider_name = "ollama"
    provider.model_name = "llama3.2:1b"
    return provider


VALID_K8S_YAML = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  namespace: default
spec:
  replicas: 2
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp
        image: myapp:1.0
        ports:
        - name: http
          containerPort: 8080
        resources:
          limits:
            cpu: 500m
---
apiVersion: v1
kind: Service
metadata:
  name: myapp
  namespace: default
spec:
  type: ClusterIP
  selector:
    app: myapp
  ports:
  - port: 80
    targetPort: 8080
"""


def test_valid_manifest_passes_all_checks(mock_provider):
    mock_provider.generate.return_value = VALID_K8S_YAML
    generator = K8sManifestGenerator(provider=mock_provider)
    result = generator.generate(app_name="myapp", image="myapp:1.0", port=8080, replicas=2, namespace="default")
    assert result.is_valid is True
    assert len(result.documents) == 2
    assert result.provider == "ollama"


def test_reject_duplicate_yaml_mapping_keys(mock_provider):
    yaml_with_duplicate_keys = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  namespace: default
spec:
  replicas: 1
  replicas: 2
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp
        image: myapp:1.0
        ports:
        - containerPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: myapp
  namespace: default
spec:
  selector:
    app: myapp
  ports:
  - port: 8080
"""
    mock_provider.generate.return_value = yaml_with_duplicate_keys
    generator = K8sManifestGenerator(provider=mock_provider)
    with pytest.raises(DevOpsAIError) as exc_info:
        generator.generate(app_name="myapp", image="myapp:1.0", port=8080)
    assert exc_info.value.code == "GENERATOR_VALIDATION_ERROR"
    assert "duplicate" in str(exc_info.value).lower()


def test_reject_scalar_or_list_yaml(mock_provider):
    generator = K8sManifestGenerator(provider=mock_provider)

    mock_provider.generate.return_value = "just a plain scalar string"
    with pytest.raises(DevOpsAIError) as exc_info:
        generator.generate(app_name="myapp", image="myapp:1.0", port=8080)
    assert exc_info.value.code == "GENERATOR_VALIDATION_ERROR"

    mock_provider.generate.return_value = "- item 1\n- item 2"
    with pytest.raises(DevOpsAIError) as exc_info2:
        generator.generate(app_name="myapp", image="myapp:1.0", port=8080)
    assert exc_info2.value.code == "GENERATOR_VALIDATION_ERROR"


def test_require_exactly_one_deployment_and_service(mock_provider):
    generator = K8sManifestGenerator(provider=mock_provider)

    # Missing Service
    mock_provider.generate.return_value = VALID_K8S_YAML.split("---")[0]
    with pytest.raises(DevOpsAIError) as exc_info:
        generator.generate(app_name="myapp", image="myapp:1.0", port=8080, replicas=2)
    assert "Expected exactly one Service" in str(exc_info.value)

    # Extra ConfigMap
    extra_yaml = VALID_K8S_YAML + "\n---\napiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: cm\ndata:\n  k: v"
    mock_provider.generate.return_value = extra_yaml
    with pytest.raises(DevOpsAIError) as exc_info2:
        generator.generate(app_name="myapp", image="myapp:1.0", port=8080, replicas=2)
    assert "Unexpected Kubernetes resource kinds found" in str(exc_info2.value)


def test_validate_api_version_and_kind(mock_provider):
    # Invalid deployment apiVersion
    bad_api = VALID_K8S_YAML.replace("apiVersion: apps/v1", "apiVersion: v1")
    mock_provider.generate.return_value = bad_api
    generator = K8sManifestGenerator(provider=mock_provider)
    with pytest.raises(DevOpsAIError) as exc_info:
        generator.generate(app_name="myapp", image="myapp:1.0", port=8080, replicas=2)
    assert "must have apiVersion 'apps/v1'" in str(exc_info.value)


def test_validate_metadata_types_and_dns_names(mock_provider):
    generator = K8sManifestGenerator(provider=mock_provider)

    # Invalid DNS name (uppercase)
    bad_name = VALID_K8S_YAML.replace("name: myapp", "name: MyApp_Invalid")
    mock_provider.generate.return_value = bad_name
    with pytest.raises(DevOpsAIError) as exc_info:
        generator.generate(app_name="myapp", image="myapp:1.0", port=8080, replicas=2)
    assert "not a valid DNS subdomain name" in str(exc_info.value)


def test_reject_boolean_replicas_and_ports(mock_provider):
    generator = K8sManifestGenerator(provider=mock_provider)

    # Boolean replicas in CLI
    with pytest.raises(DevOpsAIError) as exc_info:
        generator.generate(app_name="myapp", image="myapp:1.0", port=8080, replicas=True)  # type: ignore[arg-type]
    assert exc_info.value.code == "INVALID_ARGUMENT"
    assert "booleans rejected" in str(exc_info.value)

    # Boolean port in CLI
    with pytest.raises(DevOpsAIError) as exc_info2:
        generator.generate(app_name="myapp", image="myapp:1.0", port=False)  # type: ignore[arg-type]
    assert exc_info2.value.code == "INVALID_ARGUMENT"
    assert "booleans rejected" in str(exc_info2.value)


def test_support_named_target_port(mock_provider):
    named_yaml = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp
        image: myapp:1.0
        ports:
        - name: web-port
          containerPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: myapp
  namespace: default
spec:
  selector:
    app: myapp
  ports:
  - port: 80
    targetPort: web-port
"""
    mock_provider.generate.return_value = named_yaml
    generator = K8sManifestGenerator(provider=mock_provider)
    result = generator.generate(app_name="myapp", image="myapp:1.0", port=8080, replicas=1)
    assert result.is_valid is True


def test_target_port_omitted_defaults_to_service_port(mock_provider):
    # Omitted targetPort should default to Service port (8080)
    omitted_target = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp
        image: myapp:1.0
        ports:
        - containerPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: myapp
  namespace: default
spec:
  selector:
    app: myapp
  ports:
  - port: 8080
"""
    mock_provider.generate.return_value = omitted_target
    generator = K8sManifestGenerator(provider=mock_provider)
    result = generator.generate(app_name="myapp", image="myapp:1.0", port=8080, replicas=1)
    assert result.is_valid is True


def test_target_port_container_port_mismatch_fails(mock_provider):
    mismatch_yaml = VALID_K8S_YAML.replace("targetPort: 8080", "targetPort: 9999")
    mock_provider.generate.return_value = mismatch_yaml
    generator = K8sManifestGenerator(provider=mock_provider)
    with pytest.raises(DevOpsAIError) as exc_info:
        generator.generate(app_name="myapp", image="myapp:1.0", port=8080, replicas=2)
    assert "does not match any Deployment containerPort" in str(exc_info.value)


def test_node_port_optional_for_nodeport_service(mock_provider):
    # Service type NodePort without explicit nodePort passes (Kubernetes auto-allocates)
    nodeport_yaml = VALID_K8S_YAML.replace("type: ClusterIP", "type: NodePort")
    mock_provider.generate.return_value = nodeport_yaml
    generator = K8sManifestGenerator(provider=mock_provider)
    result = generator.generate(app_name="myapp", image="myapp:1.0", port=8080, replicas=2, service_type="NodePort")
    assert result.is_valid is True


def test_node_port_explicit_range_enforced(mock_provider):
    # NodePort < 30000 fails
    bad_nodeport = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  namespace: default
spec:
  replicas: 2
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp
        image: myapp:1.0
        ports:
        - containerPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: myapp
  namespace: default
spec:
  type: NodePort
  selector:
    app: myapp
  ports:
  - port: 80
    targetPort: 8080
    nodePort: 25000
"""
    mock_provider.generate.return_value = bad_nodeport
    generator = K8sManifestGenerator(provider=mock_provider)
    with pytest.raises(DevOpsAIError) as exc_info:
        generator.generate(app_name="myapp", image="myapp:1.0", port=8080, replicas=2, service_type="NodePort")
    assert "outside standard Kubernetes default range 30000-32767" in str(exc_info.value)


def test_node_port_rejected_for_cluster_ip(mock_provider):
    # NodePort specified on ClusterIP fails
    bad_clusterip = VALID_K8S_YAML.replace("targetPort: 8080", "targetPort: 8080\n    nodePort: 31000")
    mock_provider.generate.return_value = bad_clusterip
    generator = K8sManifestGenerator(provider=mock_provider)
    with pytest.raises(DevOpsAIError) as exc_info:
        generator.generate(app_name="myapp", image="myapp:1.0", port=8080, replicas=2, service_type="ClusterIP")
    assert "must not specify nodePort" in str(exc_info.value)


def test_namespace_validation(mock_provider):
    mismatch_ns = VALID_K8S_YAML.replace("namespace: default", "namespace: prod")
    mock_provider.generate.return_value = mismatch_ns
    generator = K8sManifestGenerator(provider=mock_provider)
    with pytest.raises(DevOpsAIError) as exc_info:
        generator.generate(app_name="myapp", image="myapp:1.0", port=8080, replicas=2, namespace="default")
    assert "Namespace mismatch" in str(exc_info.value)


def test_k8s_refuse_overwrite_without_force(mock_provider, tmp_path: Path):
    target = tmp_path / "k8s.yaml"
    target.write_text("EXISTING MANIFEST", encoding="utf-8")

    mock_provider.generate.return_value = VALID_K8S_YAML
    generator = K8sManifestGenerator(provider=mock_provider)
    with pytest.raises(DevOpsAIError) as exc_info:
        generator.generate(app_name="myapp", image="myapp:1.0", port=8080, replicas=2, output_path=target, force=False)
    assert exc_info.value.code == "FILE_ALREADY_EXISTS"
    assert target.read_text(encoding="utf-8") == "EXISTING MANIFEST"


def test_k8s_atomic_overwrite_with_force(mock_provider, tmp_path: Path):
    target = tmp_path / "k8s.yaml"
    target.write_text("EXISTING MANIFEST", encoding="utf-8")

    mock_provider.generate.return_value = VALID_K8S_YAML
    generator = K8sManifestGenerator(provider=mock_provider)
    result = generator.generate(app_name="myapp", image="myapp:1.0", port=8080, replicas=2, output_path=target, force=True)
    assert result.is_valid is True
    assert "kind: Deployment" in target.read_text(encoding="utf-8")
