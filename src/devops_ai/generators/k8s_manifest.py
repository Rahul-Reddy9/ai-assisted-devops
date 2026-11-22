# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""AI-powered Kubernetes manifest generator with PyYAML structural validation."""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Literal, Optional
import yaml
from devops_ai.core.exceptions import DevOpsAIError
from devops_ai.core.file_io import atomic_write_text
from devops_ai.core.logging import get_logger
from devops_ai.llm.base import BaseLLMProvider
from devops_ai.llm.router import GenerationResult, LLMRouter

logger = get_logger("generators.k8s")

K8S_SYSTEM_PROMPT = """You are a Senior Kubernetes Architect.
Your task is to generate complete, valid, production-grade Kubernetes YAML manifests for the requested application.

Strict output rules:
1. Output ONLY the raw Kubernetes YAML.
2. Separate multiple documents with '---'.
3. Do NOT enclose output in markdown backticks (no ``` or ```yaml).
4. Do NOT include explanatory conversation.

Best Practices:
- Include exactly one Deployment and one corresponding Service.
- Set matching labels and selectors (e.g. app: <app-name>).
- Include resource requests and limits (cpu and memory).
- Include basic liveness and readiness probes.
- Use explicit apiVersion (apps/v1 for Deployment, v1 for Service).
- Configure the requested Service type (ClusterIP, NodePort, or LoadBalancer).
"""

ServiceType = Literal["ClusterIP", "NodePort", "LoadBalancer"]


class UniqueKeySafeLoader(yaml.SafeLoader):
    """SafeLoader that detects and rejects duplicate YAML mapping keys."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> Dict[Any, Any]:
        mapping: Dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key '{key}'",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


@dataclass
class K8sValidationResult:
    """Result of Kubernetes manifest validation."""
    is_valid: bool
    content: str
    documents: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    fallback_used: bool = False


class K8sManifestGenerator:
    """Generates validated Kubernetes manifests using LLMs."""

    def __init__(self, provider: Optional[BaseLLMProvider] = None) -> None:
        self.provider = provider

    def generate(
        self,
        app_name: str,
        image: str,
        port: int = 8080,
        replicas: int = 1,
        namespace: str = "default",
        service_type: ServiceType = "ClusterIP",
        output_path: Optional[Path] = None,
        force: bool = False,
        provider_name: Optional[str] = None,
        model: Optional[str] = None,
        host: Optional[str] = None,
        timeout: Optional[float] = None,
        allow_fallback: Optional[bool] = None,
    ) -> K8sValidationResult:
        if type(port) is not int or isinstance(port, bool) or port < 1 or port > 65535:
            raise DevOpsAIError(
                f"Invalid port: {port}. Port must be a strict integer between 1 and 65535 (booleans rejected).",
                code="INVALID_ARGUMENT",
            )
        if type(replicas) is not int or isinstance(replicas, bool) or replicas < 1:
            raise DevOpsAIError(
                f"Replicas must be a positive integer >= 1, got: {replicas} (booleans rejected).",
                code="INVALID_ARGUMENT",
            )
        if service_type not in ["ClusterIP", "NodePort", "LoadBalancer"]:
            raise DevOpsAIError(
                f"Unsupported service type: '{service_type}'. Must be ClusterIP, NodePort, or LoadBalancer.",
                code="INVALID_ARGUMENT",
            )

        prompt = (
            f"Generate Kubernetes manifests for application '{app_name}'. "
            f"Container image: '{image}'. Container port: {port}. "
            f"Replicas: {replicas}. Namespace: '{namespace}'. Service type: '{service_type}'."
        )

        if self.provider:
            raw_output = self.provider.generate(
                prompt=prompt,
                system_prompt=K8S_SYSTEM_PROMPT,
                temperature=0.1,
            )
            gen_result = GenerationResult(
                content=raw_output,
                provider=self.provider.provider_name,
                model=self.provider.model_name,
                fallback_used=False,
            )
        else:
            gen_result = LLMRouter.generate_with_fallback(
                prompt=prompt,
                system_prompt=K8S_SYSTEM_PROMPT,
                provider_name=provider_name,
                model=model,
                host=host,
                timeout=timeout,
                allow_fallback=allow_fallback,
                temperature=0.1,
            )

        cleaned_content = self._clean_markdown_fences(gen_result.content)
        validation_result = self._validate_k8s_yaml(
            cleaned_content,
            expected_port=port,
            expected_replicas=replicas,
            expected_service_type=service_type,
            expected_namespace=namespace,
            expected_app_name=app_name,
        )
        validation_result.provider = gen_result.provider
        validation_result.model = gen_result.model
        validation_result.fallback_used = gen_result.fallback_used

        if output_path:
            atomic_write_text(output_path, cleaned_content, force=force)
            logger.info(f"Saved generated Kubernetes manifest to {output_path}")

        return validation_result

    @staticmethod
    def _clean_markdown_fences(text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:yaml|yml)?\r?\n", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\r?\n```$", "", cleaned)
        return cleaned.strip()

    @staticmethod
    def _validate_k8s_yaml(
        content: str,
        expected_port: int,
        expected_replicas: int,
        expected_service_type: str,
        expected_namespace: str,
        expected_app_name: Optional[str] = None,
    ) -> K8sValidationResult:
        """Validate Kubernetes manifests against structural, schema, and consistency rules."""
        try:
            raw_docs = list(yaml.load_all(content, Loader=UniqueKeySafeLoader))
        except yaml.YAMLError as e:
            raise DevOpsAIError(f"Generated output is not valid YAML: {e}", code="GENERATOR_VALIDATION_ERROR") from None

        # 1. Document cardinality and null/scalar checks
        if not raw_docs:
            raise DevOpsAIError("Generated Kubernetes manifest contains no YAML documents.", code="GENERATOR_VALIDATION_ERROR")

        docs: List[Dict[str, Any]] = []
        for idx, doc in enumerate(raw_docs, 1):
            if doc is None:
                raise DevOpsAIError(f"YAML document #{idx} is null/empty.", code="GENERATOR_VALIDATION_ERROR")
            if not isinstance(doc, dict):
                raise DevOpsAIError(
                    f"YAML document #{idx} is not a valid Kubernetes mapping (got {type(doc).__name__}).",
                    code="GENERATOR_VALIDATION_ERROR",
                )
            docs.append(doc)

        deployments = [d for d in docs if d.get("kind") == "Deployment"]
        services = [d for d in docs if d.get("kind") == "Service"]

        if len(deployments) != 1:
            raise DevOpsAIError(
                f"Expected exactly one Deployment manifest, found {len(deployments)}.",
                code="GENERATOR_VALIDATION_ERROR",
            )
        if len(services) != 1:
            raise DevOpsAIError(
                f"Expected exactly one Service manifest, found {len(services)}.",
                code="GENERATOR_VALIDATION_ERROR",
            )
        if len(docs) != 2:
            extra_kinds = [d.get("kind") for d in docs if d.get("kind") not in ("Deployment", "Service")]
            raise DevOpsAIError(
                f"Unexpected Kubernetes resource kinds found: {extra_kinds}. Expected only Deployment and Service.",
                code="GENERATOR_VALIDATION_ERROR",
            )

        deployment_doc = deployments[0]
        service_doc = services[0]

        # 2. Validate apiVersion and kind
        if deployment_doc.get("apiVersion") != "apps/v1":
            raise DevOpsAIError(
                f"Deployment must have apiVersion 'apps/v1', got '{deployment_doc.get('apiVersion')}'.",
                code="GENERATOR_VALIDATION_ERROR",
            )
        if service_doc.get("apiVersion") != "v1":
            raise DevOpsAIError(
                f"Service must have apiVersion 'v1', got '{service_doc.get('apiVersion')}'.",
                code="GENERATOR_VALIDATION_ERROR",
            )

        # 3. Metadata validation
        dep_meta = deployment_doc.get("metadata")
        svc_meta = service_doc.get("metadata")
        if not isinstance(dep_meta, dict):
            raise DevOpsAIError("Deployment 'metadata' must be a mapping dictionary.", code="GENERATOR_VALIDATION_ERROR")
        if not isinstance(svc_meta, dict):
            raise DevOpsAIError("Service 'metadata' must be a mapping dictionary.", code="GENERATOR_VALIDATION_ERROR")

        dep_name = dep_meta.get("name")
        svc_name = svc_meta.get("name")
        if not isinstance(dep_name, str) or not dep_name.strip():
            raise DevOpsAIError("Deployment metadata 'name' must be a non-empty string.", code="GENERATOR_VALIDATION_ERROR")
        if not isinstance(svc_name, str) or not svc_name.strip():
            raise DevOpsAIError("Service metadata 'name' must be a non-empty string.", code="GENERATOR_VALIDATION_ERROR")

        # Validate RFC 1123 DNS subdomain format
        dns_regex = r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$"
        if not re.match(dns_regex, dep_name):
            raise DevOpsAIError(f"Deployment name '{dep_name}' is not a valid DNS subdomain name.", code="GENERATOR_VALIDATION_ERROR")
        if not re.match(dns_regex, svc_name):
            raise DevOpsAIError(f"Service name '{svc_name}' is not a valid DNS subdomain name.", code="GENERATOR_VALIDATION_ERROR")

        dep_ns = dep_meta.get("namespace")
        svc_ns = svc_meta.get("namespace")
        if dep_ns is not None and not isinstance(dep_ns, str):
            raise DevOpsAIError("Deployment namespace must be a string.", code="GENERATOR_VALIDATION_ERROR")
        if svc_ns is not None and not isinstance(svc_ns, str):
            raise DevOpsAIError("Service namespace must be a string.", code="GENERATOR_VALIDATION_ERROR")

        # Namespace policy: if expected_namespace is requested, both must match
        if dep_ns != expected_namespace or svc_ns != expected_namespace:
            raise DevOpsAIError(
                f"Namespace mismatch: expected '{expected_namespace}', Deployment has '{dep_ns}', Service has '{svc_ns}'.",
                code="GENERATOR_VALIDATION_ERROR",
            )

        # 4. Deployment spec, selector, and template types
        dep_spec = deployment_doc.get("spec")
        if not isinstance(dep_spec, dict):
            raise DevOpsAIError("Deployment 'spec' must be a dictionary.", code="GENERATOR_VALIDATION_ERROR")

        selector = dep_spec.get("selector")
        if not isinstance(selector, dict):
            raise DevOpsAIError("Deployment 'spec.selector' must be a dictionary.", code="GENERATOR_VALIDATION_ERROR")

        match_labels = selector.get("matchLabels")
        if not isinstance(match_labels, dict) or len(match_labels) == 0:
            raise DevOpsAIError("Deployment must have non-empty 'spec.selector.matchLabels'.", code="GENERATOR_VALIDATION_ERROR")

        template = dep_spec.get("template")
        if not isinstance(template, dict):
            raise DevOpsAIError("Deployment 'spec.template' must be a dictionary.", code="GENERATOR_VALIDATION_ERROR")

        template_metadata = template.get("metadata")
        if not isinstance(template_metadata, dict):
            raise DevOpsAIError("Deployment 'spec.template.metadata' must be a dictionary.", code="GENERATOR_VALIDATION_ERROR")

        template_labels = template_metadata.get("labels")
        if not isinstance(template_labels, dict) or len(template_labels) == 0:
            raise DevOpsAIError("Deployment must have non-empty 'spec.template.metadata.labels'.", code="GENERATOR_VALIDATION_ERROR")

        # Label consistency & type validation
        for k, v in match_labels.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise DevOpsAIError(f"Deployment matchLabels '{k}: {v}' must have string keys and values.", code="GENERATOR_VALIDATION_ERROR")
            if template_labels.get(k) != v:
                raise DevOpsAIError(
                    f"Deployment selector mismatch: matchLabels '{k}: {v}' does not match template label '{template_labels.get(k)}'.",
                    code="GENERATOR_VALIDATION_ERROR",
                )

        for k, v in template_labels.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise DevOpsAIError(f"Pod template label '{k}: {v}' must have string keys and values.", code="GENERATOR_VALIDATION_ERROR")

        # 5. Service spec and selector compatibility
        svc_spec = service_doc.get("spec")
        if not isinstance(svc_spec, dict):
            raise DevOpsAIError("Service 'spec' must be a dictionary.", code="GENERATOR_VALIDATION_ERROR")

        svc_selector = svc_spec.get("selector")
        if not isinstance(svc_selector, dict) or len(svc_selector) == 0:
            raise DevOpsAIError("Service must have a non-empty 'spec.selector' dictionary.", code="GENERATOR_VALIDATION_ERROR")
        for k, v in svc_selector.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise DevOpsAIError(f"Service selector '{k}: {v}' must have string keys and values.", code="GENERATOR_VALIDATION_ERROR")
            if template_labels.get(k) != v:
                raise DevOpsAIError(
                    f"Service selector mismatch: '{k}: {v}' does not match Pod template labels.",
                    code="GENERATOR_VALIDATION_ERROR",
                )

        # 6. Container specifications
        template_spec = template.get("spec")
        if not isinstance(template_spec, dict):
            raise DevOpsAIError("Deployment 'spec.template.spec' must be a dictionary.", code="GENERATOR_VALIDATION_ERROR")

        containers = template_spec.get("containers")
        if not isinstance(containers, list) or len(containers) == 0:
            raise DevOpsAIError("Deployment must specify a non-empty list of 'containers'.", code="GENERATOR_VALIDATION_ERROR")

        container_numeric_ports: List[int] = []
        container_named_ports: List[str] = []

        rfc6335_port_name_regex = r"^[a-z0-9]([a-z0-9-]{0,13}[a-z0-9])?$"

        for idx, c in enumerate(containers):
            if not isinstance(c, dict):
                raise DevOpsAIError(f"Container #{idx+1} must be a dictionary mapping.", code="GENERATOR_VALIDATION_ERROR")
            c_name = c.get("name")
            c_image = c.get("image")
            if not isinstance(c_name, str) or not c_name.strip():
                raise DevOpsAIError(f"Container #{idx+1} must have a valid non-empty 'name'.", code="GENERATOR_VALIDATION_ERROR")
            if not isinstance(c_image, str) or not c_image.strip():
                raise DevOpsAIError(f"Container '{c_name}' must have a valid non-empty 'image'.", code="GENERATOR_VALIDATION_ERROR")

            ports = c.get("ports")
            container_local_port_names = set()
            if ports is not None:
                if not isinstance(ports, list):
                    raise DevOpsAIError(f"Container '{c_name}' ports must be a list.", code="GENERATOR_VALIDATION_ERROR")
                for p in ports:
                    if not isinstance(p, dict):
                        raise DevOpsAIError(f"Container '{c_name}' port entry must be a dictionary.", code="GENERATOR_VALIDATION_ERROR")
                    cp = p.get("containerPort")
                    if cp is None:
                        raise DevOpsAIError(f"Container '{c_name}' port entry missing 'containerPort'.", code="GENERATOR_VALIDATION_ERROR")
                    if type(cp) is not int or isinstance(cp, bool) or cp < 1 or cp > 65535:
                        raise DevOpsAIError(
                            f"Container '{c_name}' containerPort must be a strict integer between 1 and 65535 (booleans rejected), got {cp}.",
                            code="GENERATOR_VALIDATION_ERROR",
                        )
                    container_numeric_ports.append(cp)

                    p_name = p.get("name")
                    if p_name is not None:
                        if not isinstance(p_name, str) or not re.match(rfc6335_port_name_regex, p_name):
                            raise DevOpsAIError(
                                f"Container '{c_name}' port name '{p_name}' is invalid. Must be RFC 6335 compatible (1-15 lowercase alphanumeric/hyphen).",
                                code="GENERATOR_VALIDATION_ERROR",
                            )
                        if p_name in container_local_port_names:
                            raise DevOpsAIError(
                                f"Duplicate port name '{p_name}' detected in container '{c_name}'.",
                                code="GENERATOR_VALIDATION_ERROR",
                            )
                        container_local_port_names.add(p_name)
                        container_named_ports.append(p_name)

        if not container_numeric_ports:
            raise DevOpsAIError(
                "Deployment containers must specify 'ports' with valid integer 'containerPort'.",
                code="GENERATOR_VALIDATION_ERROR",
            )

        if expected_port not in container_numeric_ports:
            raise DevOpsAIError(
                f"Generated containerPort(s) {container_numeric_ports} do not include requested port {expected_port}.",
                code="GENERATOR_VALIDATION_ERROR",
            )

        # 7. Service ports validation
        svc_ports = svc_spec.get("ports")
        if not isinstance(svc_ports, list) or len(svc_ports) == 0:
            raise DevOpsAIError("Service must specify a non-empty 'spec.ports' list.", code="GENERATOR_VALIDATION_ERROR")

        for idx, sp in enumerate(svc_ports):
            if not isinstance(sp, dict):
                raise DevOpsAIError(f"Service port entry #{idx+1} must be a dictionary.", code="GENERATOR_VALIDATION_ERROR")

            s_port = sp.get("port")
            if s_port is None or type(s_port) is not int or isinstance(s_port, bool) or s_port < 1 or s_port > 65535:
                raise DevOpsAIError(
                    f"Service port #{idx+1} 'port' must be a strict integer between 1 and 65535 (booleans rejected), got {s_port}.",
                    code="GENERATOR_VALIDATION_ERROR",
                )

            # targetPort defaulting: If omitted, Kubernetes defaults targetPort to the Service's port
            target_port = sp.get("targetPort")
            if target_port is None:
                effective_tp = s_port
            else:
                effective_tp = target_port

            if type(effective_tp) is int and not isinstance(effective_tp, bool):
                if effective_tp not in container_numeric_ports:
                    raise DevOpsAIError(
                        f"Service targetPort '{effective_tp}' does not match any Deployment containerPort: {container_numeric_ports}.",
                        code="GENERATOR_VALIDATION_ERROR",
                    )
            elif isinstance(effective_tp, str):
                if not re.match(rfc6335_port_name_regex, effective_tp):
                    raise DevOpsAIError(
                        f"Service named targetPort '{effective_tp}' is not a valid port name.",
                        code="GENERATOR_VALIDATION_ERROR",
                    )
                if effective_tp not in container_named_ports:
                    raise DevOpsAIError(
                        f"Service named targetPort '{effective_tp}' does not match any container port name: {container_named_ports}.",
                        code="GENERATOR_VALIDATION_ERROR",
                    )
            else:
                raise DevOpsAIError(
                    f"Invalid Service targetPort value '{effective_tp}' (type {type(effective_tp).__name__}). Must be an integer or port name string.",
                    code="GENERATOR_VALIDATION_ERROR",
                )

            proto = sp.get("protocol")
            if proto is not None and proto not in ("TCP", "UDP", "SCTP"):
                raise DevOpsAIError(
                    f"Service port #{idx+1} protocol '{proto}' is invalid. Supported: TCP, UDP, SCTP.",
                    code="GENERATOR_VALIDATION_ERROR",
                )

        # 8. Replicas validation
        replicas = dep_spec.get("replicas")
        if replicas is not None:
            if type(replicas) is not int or isinstance(replicas, bool) or replicas < 1:
                raise DevOpsAIError(
                    f"Deployment replicas must be a strict integer >= 1 (booleans rejected), got: {replicas}.",
                    code="GENERATOR_VALIDATION_ERROR",
                )
            if replicas != expected_replicas:
                raise DevOpsAIError(
                    f"Generated replicas ({replicas}) does not match requested replicas ({expected_replicas}).",
                    code="GENERATOR_VALIDATION_ERROR",
                )

        # 9. Service Type & NodePort validation
        svc_type = svc_spec.get("type", "ClusterIP")
        if svc_type not in ["ClusterIP", "NodePort", "LoadBalancer"]:
            raise DevOpsAIError(f"Unsupported Service type '{svc_type}'.", code="GENERATOR_VALIDATION_ERROR")
        if svc_type != expected_service_type:
            raise DevOpsAIError(
                f"Generated Service type '{svc_type}' does not match requested service type '{expected_service_type}'.",
                code="GENERATOR_VALIDATION_ERROR",
            )

        if expected_service_type == "NodePort":
            for sp in svc_ports:
                node_port = sp.get("nodePort")
                if node_port is not None:
                    if type(node_port) is not int or isinstance(node_port, bool) or node_port < 30000 or node_port > 32767:
                        raise DevOpsAIError(
                            f"Explicit nodePort {node_port} is outside standard Kubernetes default range 30000-32767 (booleans rejected).",
                            code="GENERATOR_VALIDATION_ERROR",
                        )
        else:
            for sp in svc_ports:
                if sp.get("nodePort") is not None:
                    raise DevOpsAIError(
                        f"Service of type '{svc_type}' must not specify nodePort.",
                        code="GENERATOR_VALIDATION_ERROR",
                    )

        # 10. Heuristic recommendations
        warnings: List[str] = []
        for c in containers:
            resources = c.get("resources")
            if not isinstance(resources, dict) or not isinstance(resources.get("limits"), dict):
                warnings.append(f"Container '{c.get('name', 'app')}' has no resource limits specified.")
            sec_ctx = c.get("securityContext")
            if not isinstance(sec_ctx, dict):
                warnings.append(f"Container '{c.get('name', 'app')}' has no securityContext specified.")
            if not c.get("livenessProbe"):
                warnings.append(f"Container '{c.get('name', 'app')}' has no livenessProbe specified.")
            if not c.get("readinessProbe"):
                warnings.append(f"Container '{c.get('name', 'app')}' has no readinessProbe specified.")

        for w in warnings:
            logger.warning(f"Kubernetes validation warning: {w}")

        return K8sValidationResult(
            is_valid=True,
            content=content,
            documents=docs,
            warnings=warnings,
        )
