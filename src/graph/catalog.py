"""
Knowledge Graph del Nomenclador usando NetworkX.

- El grafo vive en memoria durante la sesion (cache de lectura)
- Persistencia dual: PostgreSQL (ACID, concurrencia) + JSON (fallback local)
- Gap B: patrón repositorio con write-through a Supabase
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)

from .schema import (
    NodeType, EdgeType,
    ConceptNode, FieldNode, ClassifierNode,
    OperationNode, ContextNode, SourceNode,
    NormativeNode, AnonymizationRuleNode,
    DataClassification, ReviewStatus,
    QualityIssueNode, IssueSeverity,
)


def _get_db_connection():
    """Obtener conexion PostgreSQL si DATABASE_URL esta configurada.

    Returns psycopg connection o None si no hay configuracion.
    """
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        return None
    try:
        import psycopg
        return psycopg.connect(db_url)
    except ImportError:
        return None
    except Exception as e:
        logger.warning("PostgreSQL connection fallo: %s — usando modo JSON fallback", e)
        return None


class NomencladorGraph:
    """Knowledge Graph del nomenclador institucional.

    Modo dual (Gap B):
    - Si DATABASE_URL esta configurada: write-through a PostgreSQL + NetworkX cache
    - Si no: fallback a JSON local (modo legacy)
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.version = "1.0.0"
        self.version_history = []
        self._db = _get_db_connection()
        self._loaded_mtime: float = 0.0

    def _db_upsert_node(self, node_id: str, node_data: dict):
        """Write-through: persistir nodo en PostgreSQL (Gap B)."""
        if not self._db:
            return
        node_type = node_data.get("type", "concept")
        try:
            with self._db.cursor() as cur:
                cur.execute(
                    "INSERT INTO governance.graph_nodes (id, type, data) VALUES (%s, %s, %s) "
                    "ON CONFLICT (id) DO UPDATE SET type = EXCLUDED.type, data = EXCLUDED.data",
                    (node_id, node_type, json.dumps(node_data, ensure_ascii=False, default=str))
                )
            self._db.commit()
        except Exception as e:
            logger.warning("PostgreSQL upsert_node fallo para %s: %s", node_id, e)
            try:
                self._db.rollback()
            except Exception:
                pass

    def _db_upsert_edge(self, source: str, target: str, edge_type: str, edge_data: dict = None):
        """Write-through: persistir arista en PostgreSQL (Gap B)."""
        if not self._db:
            return
        try:
            with self._db.cursor() as cur:
                cur.execute(
                    "INSERT INTO governance.graph_edges (source_id, target_id, type, data) VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (source_id, target_id, type) DO UPDATE SET data = EXCLUDED.data",
                    (source, target, edge_type, json.dumps(edge_data or {}, ensure_ascii=False, default=str))
                )
            self._db.commit()
        except Exception as e:
            logger.warning("PostgreSQL upsert_edge fallo para %s->%s: %s", source, target, e)
            try:
                self._db.rollback()
            except Exception:
                pass

    def _db_delete_node(self, node_id: str):
        """Eliminar nodo de PostgreSQL (Gap B)."""
        if not self._db:
            return
        try:
            with self._db.cursor() as cur:
                cur.execute("DELETE FROM governance.graph_nodes WHERE id = %s", (node_id,))
            self._db.commit()
        except Exception as e:
            logger.warning("PostgreSQL delete_node fallo para %s: %s", node_id, e)
            try:
                self._db.rollback()
            except Exception:
                pass

    def _db_load_all(self) -> dict:
        """Cargar todo el grafo desde PostgreSQL (Gap B).

        Returns dict con nodes, edges, version o vacio si no hay datos.
        """
        if not self._db:
            return {}
        try:
            with self._db.cursor() as cur:
                cur.execute("SELECT id, type, data FROM governance.graph_nodes")
                nodes = []
                for row in cur.fetchall():
                    node_data = row[2] if isinstance(row[2], dict) else json.loads(row[2] or "{}")
                    node_data["id"] = row[0]
                    nodes.append(node_data)

                cur.execute("SELECT source_id, target_id, type, data FROM governance.graph_edges")
                edges = []
                for row in cur.fetchall():
                    edge_data = row[3] if isinstance(row[3], dict) else json.loads(row[3] or "{}")
                    edges.append({"source": row[0], "target": row[1], "type": row[2], **edge_data})

                cur.execute("SELECT version FROM governance.nomenclador_version ORDER BY changed_at DESC LIMIT 1")
                version_row = cur.fetchone()
                version = version_row[0] if version_row else "1.0.0"

                return {"nodes": nodes, "links": edges, "_version": version}
        except Exception as e:
            logger.warning("PostgreSQL load_all fallo: %s", e)
            return {}

    def _db_save_version(self):
        """Persistir version actual en PostgreSQL (Gap B)."""
        if not self._db:
            return
        try:
            with self._db.cursor() as cur:
                cur.execute(
                    "INSERT INTO governance.nomenclador_version (version, total_nodes, total_edges, reason) "
                    "VALUES (%s, %s, %s, %s)",
                    (self.version, self.graph.number_of_nodes(), self.graph.number_of_edges(), "")
                )
            self._db.commit()
        except Exception as e:
            logger.warning("PostgreSQL save_version fallo: %s", e)
            try:
                self._db.rollback()
            except Exception:
                pass

    # === VERSIONADO ===

    def bump_version(self, change_type: str = "patch", reason: str = "") -> str:
        """Incrementar version del nomenclador (semantic versioning).

        Args:
            change_type: major | minor | patch
            reason: por que del cambio

        Returns:
            Nueva version
        """
        parts = self.version.split(".")
        while len(parts) < 3:
            parts.append("0")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

        if change_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif change_type == "minor":
            minor += 1
            patch = 0
        else:
            patch += 1

        old_version = self.version
        self.version = f"{major}.{minor}.{patch}"

        self.version_history.append({
            "from": old_version,
            "to": self.version,
            "type": change_type,
            "reason": reason,
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
        })

        return self.version

    def version_info(self) -> dict:
        """Info de version actual + historial."""
        return {
            "version": self.version,
            "total_changes": len(self.version_history),
            "history": self.version_history[-5:],  # ultimos 5
        }

    # === NODOS ===

    def add_concept(self, node: ConceptNode):
        data = node.model_dump()
        self.graph.add_node(node.id, **data)
        self._db_upsert_node(node.id, data)

    def add_field(self, node: FieldNode):
        data = node.model_dump()
        self.graph.add_node(node.id, **data)
        self._db_upsert_node(node.id, data)

    def add_classifier(self, node: ClassifierNode):
        data = node.model_dump()
        self.graph.add_node(node.id, **data)
        self._db_upsert_node(node.id, data)

    def add_operation(self, node: OperationNode):
        data = node.model_dump()
        self.graph.add_node(node.id, **data)
        self._db_upsert_node(node.id, data)

    def add_context(self, node: ContextNode):
        data = node.model_dump()
        self.graph.add_node(node.id, **data)
        self._db_upsert_node(node.id, data)

    def add_source(self, node: SourceNode):
        data = node.model_dump()
        self.graph.add_node(node.id, **data)
        self._db_upsert_node(node.id, data)

    # === ARISTAS ===

    def link_implementa(self, field_id: str, concept_id: str):
        """Field implementa un Concept."""
        self.graph.add_edge(field_id, concept_id, type=EdgeType.IMPLEMENTA.value)
        self._db_upsert_edge(field_id, concept_id, EdgeType.IMPLEMENTA.value)

    def link_clasificador(self, concept_id: str, classifier_id: str):
        """Concept usa un Classifier."""
        self.graph.add_edge(concept_id, classifier_id, type=EdgeType.USA_CLASIFICADOR.value)
        self._db_upsert_edge(concept_id, classifier_id, EdgeType.USA_CLASIFICADOR.value)

    def link_transforma(self, from_field: str, to_field: str, operation_id: str):
        """Field se transforma a otro Field via Operation."""
        self.graph.add_edge(from_field, operation_id, type=EdgeType.TRANSFORMA_A.value)
        self.graph.add_edge(operation_id, to_field, type=EdgeType.TRANSFORMA_A.value)
        self._db_upsert_edge(from_field, operation_id, EdgeType.TRANSFORMA_A.value)
        self._db_upsert_edge(operation_id, to_field, EdgeType.TRANSFORMA_A.value)

    def link_contexto(self, field_id: str, context_id: str):
        """Field pertenece a un Context."""
        self.graph.add_edge(field_id, context_id, type=EdgeType.PERTENECE_A.value)
        self._db_upsert_edge(field_id, context_id, EdgeType.PERTENECE_A.value)

    def link_fuente(self, field_id: str, source_id: str):
        """Field proviene de un Source."""
        self.graph.add_edge(field_id, source_id, type=EdgeType.PROVIENE_DE.value)
        self._db_upsert_edge(field_id, source_id, EdgeType.PROVIENE_DE.value)

    def link_compone(self, part_id: str, whole_id: str):
        """Concept compone otro Concept (ej: primer_nombre -> nombre_completo)."""
        self.graph.add_edge(part_id, whole_id, type=EdgeType.COMPONE.value)
        self._db_upsert_edge(part_id, whole_id, EdgeType.COMPONE.value)

    def link_deriva(self, derived_id: str, base_id: str):
        """Concept deriva de otro (ej: año_nacimiento deriva de fecha_nacimiento)."""
        self.graph.add_edge(derived_id, base_id, type=EdgeType.DERIVA_DE.value)
        self._db_upsert_edge(derived_id, base_id, EdgeType.DERIVA_DE.value)

    def add_normative(self, node: NormativeNode):
        """Agregar un documento normativo al grafo."""
        data = node.model_dump()
        self.graph.add_node(node.id, **data)
        self._db_upsert_node(node.id, data)

    def link_normative(self, concept_id: str, normative_id: str):
        """Concept esta respaldado por un NormativeDocument."""
        self.graph.add_edge(concept_id, normative_id, type=EdgeType.RESPALDADO_POR.value)
        self._db_upsert_edge(concept_id, normative_id, EdgeType.RESPALDADO_POR.value)

    # === GAP A: ANONIMIZACION ===

    def add_anonymization_rule(self, node: AnonymizationRuleNode):
        """Agregar una regla de anonimizacion al grafo."""
        data = node.model_dump()
        self.graph.add_node(node.id, **data)
        self._db_upsert_node(node.id, data)

    def link_anonymization(self, node_id: str, rule_id: str):
        """Vincular un Concept o Field con una regla de anonimizacion."""
        self.graph.add_edge(node_id, rule_id, type=EdgeType.APLICA_ANONIMIZACION.value)
        self._db_upsert_edge(node_id, rule_id, EdgeType.APLICA_ANONIMIZACION.value)

    def find_anonymization_rules(self, node_id: str) -> list[dict]:
        """Encontrar reglas de anonimizacion aplicables a un concepto o campo."""
        rules = []
        for successor in self.graph.successors(node_id):
            edge_data = self.graph.get_edge_data(node_id, successor)
            if edge_data and edge_data.get("type") == EdgeType.APLICA_ANONIMIZACION.value:
                node_data = self.graph.nodes[successor]
                rules.append({"id": successor, **node_data})
        return rules

    def set_data_classification(self, node_id: str, classification: str):
        """Establecer el nivel de clasificacion de datos de un nodo."""
        if node_id in self.graph:
            self.graph.nodes[node_id]["data_classification"] = classification
            self._db_upsert_node(node_id, dict(self.graph.nodes[node_id]))

    def get_data_classification(self, node_id: str) -> str:
        """Obtener el nivel de clasificacion de datos de un nodo."""
        if node_id in self.graph:
            return self.graph.nodes[node_id].get("data_classification", "publico")
        return "publico"

    def find_sensitive_data(self) -> list[dict]:
        """Listar todos los conceptos/campos con datos PII o sensibles."""
        results = []
        for node_id, data in self.graph.nodes(data=True):
            cls = data.get("data_classification", "publico")
            if cls in ("pii", "sensible"):
                results.append({"id": node_id, "classification": cls, **data})
        return results

    # === GAP C: WORKFLOW DE APROBACION ===

    def set_review_status(self, node_id: str, status: str, proposed_by: str = ""):
        """Establecer el estado de revision de un nodo."""
        if node_id in self.graph:
            self.graph.nodes[node_id]["review_status"] = status
            if proposed_by:
                self.graph.nodes[node_id]["proposed_by"] = proposed_by
            self._db_upsert_node(node_id, dict(self.graph.nodes[node_id]))

    def get_review_status(self, node_id: str) -> str:
        """Obtener el estado de revision de un nodo."""
        if node_id in self.graph:
            return self.graph.nodes[node_id].get("review_status", "approved")
        return "approved"

    def find_proposed_nodes(self) -> list[dict]:
        """Listar todos los nodos propuestos por IA pendientes de revision."""
        results = []
        for node_id, data in self.graph.nodes(data=True):
            status = data.get("review_status", "approved")
            if status in ("proposed", "under_review"):
                results.append({"id": node_id, "review_status": status, **data})
        return results

    def approve_node(self, node_id: str):
        """Aprobar un nodo propuesto."""
        self.set_review_status(node_id, "approved")

    def reject_node(self, node_id: str):
        """Rechazar un nodo propuesto."""
        self.set_review_status(node_id, "rejected")

    # === GAP D: CLASIFICADORES DINAMICOS ===

    def link_classifier_equivalent(self, classifier_a: str, classifier_b: str, mapping: dict = None):
        """Vincular dos clasificadores como equivalentes (ej: ICD-10 v2019 <-> ICD-10 v2024).

        Args:
            classifier_a: ID del clasificador A
            classifier_b: ID del clasificador B
            mapping: diccionario opcional {codigo_a: codigo_b}
        
        Returns:
            dict con cardinalidad del mapping y warnings si 1:N o N:1
        """
        mapping = mapping or {}
        warnings = []
        cardinality = "1:1"
        
        if mapping:
            # Analizar cardinalidad del mapping
            value_counts = {}
            for v in mapping.values():
                if isinstance(v, list):
                    cardinality = "1:N"
                    warnings.append(f"Mapping 1:N detectado: un codigo en {classifier_a} mapea a multiples en {classifier_b}")
                    break
                value_counts[str(v)] = value_counts.get(str(v), 0) + 1
            
            if cardinality == "1:1":
                # Verificar N:1 (muchos codigos A -> un codigo B)
                multi_to_one = sum(1 for c in value_counts.values() if c > 1)
                if multi_to_one > 0:
                    cardinality = "N:1"
                    warnings.append(f"Mapping N:1 detectado: {multi_to_one} codigo(s) en {classifier_a} mapean al mismo codigo en {classifier_b}")
        
        self.graph.add_edge(classifier_a, classifier_b,
                            type=EdgeType.EQUIVALE_A.value,
                            mapping=mapping,
                            cardinality=cardinality)
        self._db_upsert_edge(classifier_a, classifier_b, EdgeType.EQUIVALE_A.value, 
                             {"mapping": mapping, "cardinality": cardinality})
        
        return {"cardinality": cardinality, "warnings": warnings}

    def link_classifier_subconcept(self, child_id: str, parent_id: str):
        """Vincular un clasificador como subconcepto de otro (jerarquia).

        Ej: ICD-10 capitulo -> ICD-10 codigo especifico
        """
        self.graph.add_edge(child_id, parent_id, type=EdgeType.SUBCONCEPTO_DE.value)
        self._db_upsert_edge(child_id, parent_id, EdgeType.SUBCONCEPTO_DE.value)

    def find_classifier_equivalents(self, classifier_id: str) -> list[dict]:
        """Encontrar clasificadores equivalentes.
        
        Incluye cardinalidad del mapping y warnings si es 1:N o N:1.
        """
        results = []
        for _, target, data in self.graph.out_edges(classifier_id, data=True):
            if data.get("type") == EdgeType.EQUIVALE_A.value:
                node = self.graph.nodes[target]
                cardinality = data.get("cardinality", "1:1")
                entry = {"id": target, "mapping": data.get("mapping", {}), "cardinality": cardinality, **node}
                if cardinality != "1:1":
                    entry["warning"] = f"Cardinalidad {cardinality}: transformacion no directa, requiere revision manual"
                results.append(entry)
        for source, _, data in self.graph.in_edges(classifier_id, data=True):
            if data.get("type") == EdgeType.EQUIVALE_A.value:
                node = self.graph.nodes[source]
                cardinality = data.get("cardinality", "1:1")
                entry = {"id": source, "mapping": data.get("mapping", {}), "cardinality": cardinality, **node}
                if cardinality != "1:1":
                    entry["warning"] = f"Cardinalidad {cardinality}: transformacion no directa, requiere revision manual"
                results.append(entry)
        return results

    def find_classifier_hierarchy(self, classifier_id: str) -> dict:
        """Encontrar la jerarquia de un clasificador (padre e hijos)."""
        parents = []
        for _, target, data in self.graph.out_edges(classifier_id, data=True):
            if data.get("type") == EdgeType.SUBCONCEPTO_DE.value:
                node = self.graph.nodes[target]
                parents.append({"id": target, **node})
        children = []
        for source, _, data in self.graph.in_edges(classifier_id, data=True):
            if data.get("type") == EdgeType.SUBCONCEPTO_DE.value:
                node = self.graph.nodes[source]
                children.append({"id": source, **node})
        return {"parents": parents, "children": children}

    # === CONSULTAS ===

    def find_all_concepts(self) -> list[dict]:
        """Listar todos los conceptos canonicos."""
        results = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == NodeType.CONCEPT.value:
                results.append({"id": node_id, **data})
        return results

    def find_normative_of_concept(self, concept_id: str) -> list[dict]:
        """Encontrar todos los documentos normativos que respaldan un concepto."""
        normatives = []
        for successor in self.graph.successors(concept_id):
            edge_data = self.graph.get_edge_data(concept_id, successor)
            if edge_data and edge_data.get("type") == EdgeType.RESPALDADO_POR.value:
                node_data = self.graph.nodes[successor]
                normatives.append({"id": successor, **node_data})
        return normatives

    def find_concept(self, name: str, include_proposed: bool = False) -> Optional[dict]:
        """Buscar un concepto por nombre.
        
        Args:
            name: Nombre del concepto a buscar.
            include_proposed: Si True, incluye conceptos en estado 'proposed' o 'rejected'.
                             Por defecto solo retorna conceptos approved o under_review.
        """
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == NodeType.CONCEPT.value and data.get("name") == name:
                status = data.get("review_status", "approved")
                if not include_proposed and status in ("proposed", "rejected"):
                    continue
                return {"id": node_id, **data}
        return None

    def find_fields_of_concept(self, concept_id: str) -> list[dict]:
        """Encontrar todos los campos físicos que implementan un concepto."""
        fields = []
        for predecessor in self.graph.predecessors(concept_id):
            edge_data = self.graph.get_edge_data(predecessor, concept_id)
            if edge_data and edge_data.get("type") == EdgeType.IMPLEMENTA.value:
                node_data = self.graph.nodes[predecessor]
                fields.append({"id": predecessor, **node_data})
        return fields

    def find_classifier_of_concept(self, concept_id: str) -> Optional[dict]:
        """Encontrar el clasificador (valores validos) de un concepto."""
        for successor in self.graph.successors(concept_id):
            edge_data = self.graph.get_edge_data(concept_id, successor)
            if edge_data and edge_data.get("type") == EdgeType.USA_CLASIFICADOR.value:
                node_data = self.graph.nodes[successor]
                return {"id": successor, **node_data}
        return None

    def find_interoperability_path(self, source_db: str, target_db: str, include_proposed: bool = False) -> list[dict]:
        """
        Buscar caminos de interoperabilidad entre dos fuentes.

        Retorna lista de diccionarios con:
        - path: [field_a, concept, field_b]
        - field_a: dict del nodo field
        - field_b: dict del nodo field
        - concept: dict del nodo concept
        - classifier: dict del clasificador (si existe)

        Args:
            source_db: Nombre de la fuente origen.
            target_db: Nombre de la fuente destino.
            include_proposed: Si True, incluye conceptos en estado 'proposed' o 'rejected'.
                             Por defecto solo usa conceptos approved o under_review.
        """
        source_fields = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("type") == NodeType.FIELD.value and d.get("source_db") == source_db
        ]
        target_fields = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("type") == NodeType.FIELD.value and d.get("source_db") == target_db
        ]

        results = []
        for sf in source_fields:
            for tf in target_fields:
                for successor in self.graph.successors(sf):
                    edge = self.graph.get_edge_data(sf, successor)
                    if edge and edge.get("type") == EdgeType.IMPLEMENTA.value:
                        # Filtrar conceptos no aprobados
                        concept_status = self.graph.nodes[successor].get("review_status", "approved")
                        if not include_proposed and concept_status in ("proposed", "rejected"):
                            continue
                        for target_pred in self.graph.predecessors(successor):
                            if target_pred == tf:
                                concept_data = {"id": successor, **self.graph.nodes[successor]}
                                field_a_data = {"id": sf, **self.graph.nodes[sf]}
                                field_b_data = {"id": tf, **self.graph.nodes[tf]}

                                # Buscar clasificador del concepto
                                classifier_data = None
                                for cls in self.graph.successors(successor):
                                    cls_edge = self.graph.get_edge_data(successor, cls)
                                    if cls_edge and cls_edge.get("type") == EdgeType.USA_CLASIFICADOR.value:
                                        classifier_data = {"id": cls, **self.graph.nodes[cls]}
                                        break

                                results.append({
                                    "path": [sf, successor, tf],
                                    "field_a": field_a_data,
                                    "field_b": field_b_data,
                                    "concept": concept_data,
                                    "classifier": classifier_data,
                                })
        return results

    def list_concepts(self, include_proposed: bool = False) -> list[dict]:
        """Listar todos los conceptos canónicos.
        
        Args:
            include_proposed: Si True, incluye conceptos en estado 'proposed' o 'rejected'.
                             Por defecto solo retorna conceptos approved o under_review.
        """
        results = []
        for n, d in self.graph.nodes(data=True):
            if d.get("type") == NodeType.CONCEPT.value:
                status = d.get("review_status", "approved")
                if not include_proposed and status in ("proposed", "rejected"):
                    continue
                results.append({"id": n, **d})
        return results

    def list_fields(self, source_db: Optional[str] = None) -> list[dict]:
        """Listar campos físicos, opcionalmente filtrados por DB."""
        return [
            {"id": n, **d}
            for n, d in self.graph.nodes(data=True)
            if d.get("type") == NodeType.FIELD.value
            and (source_db is None or d.get("source_db") == source_db)
        ]

    def analyze_impact(self, concept_id: str) -> dict:
        """Analizar el impacto de cambiar un concepto.
        
        Retorna que se ve afectado si el concepto es modificado, deprecado o eliminado:
        - fields: campos fisicos que implementan este concepto
        - interop_paths: rutas de interoperabilidad que usan este concepto
        - composites: conceptos compuestos que incluyen este concepto
        - derived: conceptos que derivan de este concepto
        - classifiers: clasificadores asociados
        - normatives: documentos normativos vinculados
        - transform_operations: operaciones de transformacion que afectan fields de este concepto
        """
        impact = {
            "concept_id": concept_id,
            "fields": [],
            "interop_paths": [],
            "composites": [],
            "derived_from": [],
            "derives_to": [],
            "classifiers": [],
            "normatives": [],
            "transform_operations": [],
            "total_impact": 0,
        }
        
        if concept_id not in self.graph:
            impact["error"] = f"Concepto '{concept_id}' no encontrado en el grafo"
            return impact
        
        # 1. Fields que implementan este concepto
        fields = self.find_fields_of_concept(concept_id)
        impact["fields"] = fields
        
        # 2. Rutas de interoperabilidad (fields de este concepto conectados a otros)
        for field in fields:
            field_id = field["id"]
            field_db = field.get("source_db", "")
            # Buscar otras fuentes que interoperan con este field
            for other_field in self.list_fields():
                if other_field["id"] == field_id:
                    continue
                other_db = other_field.get("source_db", "")
                if other_db and other_db != field_db:
                    paths = self.find_interoperability_path(field_db, other_db)
                    for p in paths:
                        if p.get("concept", {}).get("id") == concept_id:
                            impact["interop_paths"].append({
                                "source_db": field_db,
                                "target_db": other_db,
                                "field_a": p["field_a"].get("id"),
                                "field_b": p["field_b"].get("id"),
                            })
        
        # 3. Conceptos compuestos (este concepto es parte de)
        for source, _, data in self.graph.in_edges(concept_id, data=True):
            if data.get("type") == EdgeType.COMPONE.value:
                node = self.graph.nodes[source]
                impact["composites"].append({"id": source, "name": node.get("name", "?")})
        
        # 4. Conceptos que derivan de este
        for source, _, data in self.graph.in_edges(concept_id, data=True):
            if data.get("type") == EdgeType.DERIVA_DE.value:
                node = self.graph.nodes[source]
                impact["derived_from"].append({"id": source, "name": node.get("name", "?")})
        
        # 5. Conceptos de los que este deriva
        for _, target, data in self.graph.out_edges(concept_id, data=True):
            if data.get("type") == EdgeType.DERIVA_DE.value:
                node = self.graph.nodes[target]
                impact["derives_to"].append({"id": target, "name": node.get("name", "?")})
        
        # 6. Clasificador asociado
        classifier = self.find_classifier_of_concept(concept_id)
        if classifier:
            impact["classifiers"].append(classifier)
        
        # 7. Normativas vinculadas
        normatives = self.find_normative_of_concept(concept_id)
        impact["normatives"] = normatives
        
        # 8. Operaciones de transformacion que afectan fields de este concepto
        for field in fields:
            for successor in self.graph.successors(field["id"]):
                edge = self.graph.get_edge_data(field["id"], successor)
                if edge and edge.get("type") == EdgeType.TRANSFORMA_A.value:
                    impact["transform_operations"].append({
                        "field": field["id"],
                        "operation": successor,
                    })
        
        impact["total_impact"] = (
            len(impact["fields"]) +
            len(impact["interop_paths"]) +
            len(impact["composites"]) +
            len(impact["derived_from"]) +
            len(impact["derives_to"]) +
            len(impact["classifiers"]) +
            len(impact["normatives"]) +
            len(impact["transform_operations"])
        )
        
        return impact

    # === VARIABLES COMPUESTAS ===

    def link_composite(self, composite_id: str, part_id: str, operation: str = "concat"):
        """Vincular un concepto compuesto con sus partes (arista COMPONE).

        Args:
            composite_id: Concepto compuesto (ej: concept:nombre_completo)
            part_id: Concepto parte (ej: concept:primer_nombre)
            operation: Tipo de composicion (concat, sum, calc, date_diff)
        """
        self.graph.add_edge(composite_id, part_id,
                            type=EdgeType.COMPONE.value, operation=operation)
        self._db_upsert_edge(composite_id, part_id, EdgeType.COMPONE.value,
                             {"operation": operation})

    def find_components(self, concept_id: str) -> list[dict]:
        """Encontrar las partes de un concepto compuesto."""
        components = []
        for _, target, data in self.graph.out_edges(concept_id, data=True):
            if data.get("type") == EdgeType.COMPONE.value:
                node_data = self.graph.nodes[target]
                components.append({"id": target, "operation": data.get("operation", ""), **node_data})
        return components

    def find_composites_of(self, part_id: str) -> list[dict]:
        """Encontrar compuestos que usan este concepto como parte."""
        composites = []
        for source, _, data in self.graph.in_edges(part_id, data=True):
            if data.get("type") == EdgeType.COMPONE.value:
                node_data = self.graph.nodes[source]
                composites.append({"id": source, "operation": data.get("operation", ""), **node_data})
        return composites

    # === CONFLICTOS DE CONTEXTO ===

    def set_context_meaning(self, concept_id: str, source_db: str, meaning: str, context: str = ""):
        """Registrar que un concepto tiene un significado distinto segun la fuente.

        Args:
            concept_id: Concepto canonico
            source_db: Base de datos donde tiene significado particular
            meaning: Significado especifico en esa fuente
            context: Contexto de negocio (ej: "ingreso hospitalario", "afiliacion seguro")
        """
        ctx_id = f"context:{concept_id}:{source_db}"
        if ctx_id not in self.graph:
            self.graph.add_node(ctx_id, type=NodeType.CONTEXT.value,
                                name=context or source_db,
                                description=meaning,
                                source_db=source_db,
                                concept_id=concept_id)
            self._db_upsert_node(ctx_id, dict(self.graph.nodes[ctx_id]))
        self.graph.add_edge(concept_id, ctx_id, type=EdgeType.TIENE_CONTEXTO.value)
        self._db_upsert_edge(concept_id, ctx_id, EdgeType.TIENE_CONTEXTO.value)

    def get_context_meanings(self, concept_id: str) -> list[dict]:
        """Obtener todos los significados contextuales de un concepto."""
        meanings = []
        for _, target, data in self.graph.out_edges(concept_id, data=True):
            if data.get("type") == EdgeType.TIENE_CONTEXTO.value:
                node = self.graph.nodes[target]
                meanings.append({"id": target, **node})
        return meanings

    def find_context_conflicts(self) -> list[dict]:
        """Encontrar conceptos con multiples significados contextuales (conflictos)."""
        conflicts = []
        for n, d in self.graph.nodes(data=True):
            if d.get("type") == NodeType.CONCEPT.value:
                meanings = self.get_context_meanings(n)
                if len(meanings) > 1:
                    conflicts.append({
                        "concept_id": n,
                        "concept_name": d.get("name", n),
                        "meanings": meanings,
                    })
        return conflicts

    # === CALIDAD DE DATOS (PMBOK Quality Management) ===

    def add_quality_issue(self, field_id: str, issue_type: str, severity: str,
                          detail: str, metric_value: float = 0.0,
                          detected_by: str = "rag_factory") -> str:
        """Registrar un issue de calidad de datos en el grafo.

        Crea un QualityIssueNode y lo vincula al FieldNode via TIENE_ISSUE.
        Retorna el ID del issue creado.
        """
        issue_id = f"issue:{field_id}:{issue_type}"
        if issue_id not in self.graph:
            self.graph.add_node(issue_id, **{
                "id": issue_id,
                "type": NodeType.QUALITY_ISSUE.value,
                "issue_type": issue_type,
                "severity": severity,
                "detail": detail,
                "metric_value": metric_value,
                "detected_by": detected_by,
            })
            self._db_upsert_node(issue_id, dict(self.graph.nodes[issue_id]))
        if not self.graph.has_edge(field_id, issue_id):
            self.graph.add_edge(field_id, issue_id, type=EdgeType.TIENE_ISSUE.value)
            self._db_upsert_edge(field_id, issue_id, EdgeType.TIENE_ISSUE.value)
        return issue_id

    def find_issues_of_field(self, field_id: str) -> list[dict]:
        """Obtener todos los issues de calidad de un campo."""
        issues = []
        for successor in self.graph.successors(field_id):
            edge = self.graph.get_edge_data(field_id, successor)
            if edge and edge.get("type") == EdgeType.TIENE_ISSUE.value:
                node = self.graph.nodes[successor]
                issues.append({"id": successor, **node})
        return issues

    def compute_quality_score(self, field_id: str) -> float:
        """Calcular quality_score de un campo basado en sus métricas.

        Fórmula ponderada:
        - completeness (40%): % no nulos
        - consistency (30%): % valores que matchean estándar
        - validity (20%): % valores que pasan formato
        - uniqueness (10%): ratio unique/total (penaliza duplicados no esperados)
        """
        node = self.graph.nodes.get(field_id)
        if not node or node.get("type") != NodeType.FIELD.value:
            return 0.0

        completeness = node.get("completeness", 0.0)
        consistency = node.get("consistency", 0.0)
        validity = node.get("validity", 0.0)
        uniqueness = node.get("uniqueness", 0.0)

        score = (completeness * 0.4 + consistency * 0.3 +
                 validity * 0.2 + uniqueness * 0.1)

        node["quality_score"] = round(score, 3)
        self._db_upsert_node(field_id, dict(node))
        return round(score, 3)

    def get_quality_summary(self, source_db: Optional[str] = None) -> dict:
        """Resumen de calidad de todas las fuentes o una específica.

        Retorna dict con:
        - total_fields: número de campos
        - avg_quality: score promedio
        - low_quality_fields: campos con quality_score < 0.5
        - issues_by_severity: {error: N, warning: N, info: N}
        """
        fields = self.list_fields(source_db)
        if not fields:
            return {"total_fields": 0, "avg_quality": 0.0,
                    "low_quality_fields": [], "issues_by_severity": {}}

        scores = []
        low_quality = []
        issues_by_severity = {"error": 0, "warning": 0, "info": 0}

        for f in fields:
            fid = f["id"]
            score = f.get("quality_score", 0.0)
            if score == 0.0:
                score = self.compute_quality_score(fid)
            scores.append(score)
            if score < 0.5:
                low_quality.append({"id": fid, "column": f.get("column", ""),
                                    "quality_score": score})

            for issue in self.find_issues_of_field(fid):
                sev = issue.get("severity", "warning")
                issues_by_severity[sev] = issues_by_severity.get(sev, 0) + 1

        return {
            "total_fields": len(fields),
            "avg_quality": round(sum(scores) / len(scores), 3) if scores else 0.0,
            "low_quality_fields": low_quality,
            "issues_by_severity": issues_by_severity,
        }

    # === STALENESS TRACKING ===

    def touch_concept(self, concept_id: str, verified_date: str = ""):
        """Actualizar last_verified de un concepto o fuente.

        Args:
            concept_id: ID del nodo a tocar
            verified_date: Fecha ISO (ej: "2026-07-09"). Si vacio, usa hoy.
        """
        if concept_id not in self.graph:
            return
        if not verified_date:
            from datetime import date
            verified_date = date.today().isoformat()
        self.graph.nodes[concept_id]["last_verified"] = verified_date
        self._db_upsert_node(concept_id, dict(self.graph.nodes[concept_id]))

    def find_stale_concepts(self, days_threshold: int = 180) -> list[dict]:
        """Encontrar conceptos/fuentes cuyo last_verified excede el umbral.

        GIGO causa #6: knowledge base desactualizada. Este metodo permite
        al agente detectar conceptos que no se han verificado en mucho tiempo.

        Args:
            days_threshold: Numero de dias para considerar stale (default 180)

        Returns:
            Lista de dicts con id, name, type, last_verified, days_since
        """
        from datetime import date, datetime
        today = date.today()
        stale = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") not in (NodeType.CONCEPT.value, NodeType.SOURCE.value, NodeType.FIELD.value):
                continue
            lv = data.get("last_verified", "")
            if not lv:
                # Sin fecha = nunca verificado = stale
                stale.append({
                    "id": node_id,
                    "name": data.get("name", node_id),
                    "type": data.get("type"),
                    "last_verified": "",
                    "days_since": -1,
                })
                continue
            try:
                verified = datetime.fromisoformat(lv).date()
                delta = (today - verified).days
                if delta > days_threshold:
                    stale.append({
                        "id": node_id,
                        "name": data.get("name", node_id),
                        "type": data.get("type"),
                        "last_verified": lv,
                        "days_since": delta,
                    })
            except (ValueError, TypeError):
                continue
        stale.sort(key=lambda x: x["days_since"], reverse=True)
        return stale

    # === CONTEXT ASSEMBLY ===

    def build_concept_context(self, concept_id: str) -> dict:
        """Ensamblar contexto completo de un concepto para el agente IA.

        Reune en un solo dict toda la informacion relevante:
        - concepto canonico (definicion, estandar, poblacion, metodologia)
        - campos que lo implementan (con quality_score, review_status, staleness)
        - clasificador asociado (valores validos)
        - normativas que lo respaldan
        - issues de calidad detectados
        - conflictos de contexto (si tiene multiples significados)
        - conceptos compuestos o derivados

        Esto permite que el agente reciba contexto estructurado en una sola
        llamada en vez de hacer N queries separadas al grafo (Context Engineering).
        """
        if concept_id not in self.graph:
            return {"error": f"Concepto '{concept_id}' no encontrado"}

        node = self.graph.nodes[concept_id]
        context = {
            "concept": {
                "id": concept_id,
                "name": node.get("name", ""),
                "definition": node.get("definition", ""),
                "standard": node.get("standard"),
                "population": node.get("population", ""),
                "capture_method": node.get("capture_method", ""),
                "custodian": node.get("custodian", ""),
                "review_status": node.get("review_status", "approved"),
                "last_verified": node.get("last_verified", ""),
                "data_classification": node.get("data_classification", "publico"),
            },
            "fields": [],
            "classifier": None,
            "normatives": [],
            "quality_issues": [],
            "context_conflicts": [],
            "composites": [],
            "derived_from": [],
        }

        # Campos que implementan este concepto
        for field in self.find_fields_of_concept(concept_id):
            f = {
                "id": field["id"],
                "source_db": field.get("source_db", ""),
                "column": field.get("column", ""),
                "quality_score": field.get("quality_score", 0.0),
                "completeness": field.get("completeness", 0.0),
                "review_status": field.get("review_status", "approved"),
                "last_verified": field.get("last_verified", ""),
            }
            # Issues de calidad del campo
            issues = self.find_issues_of_field(field["id"])
            if issues:
                f["issues"] = [{"type": i.get("issue_type"), "severity": i.get("severity"),
                                "detail": i.get("detail")} for i in issues]
            context["fields"].append(f)

        # Clasificador
        classifier = self.find_classifier_of_concept(concept_id)
        if classifier:
            context["classifier"] = {
                "id": classifier["id"],
                "name": classifier.get("name", ""),
                "standard": classifier.get("standard"),
                "version_label": classifier.get("version_label", ""),
                "is_current": classifier.get("is_current", True),
            }

        # Normativas
        normatives = self.find_normative_of_concept(concept_id)
        context["normatives"] = [
            {"id": n["id"], "title": n.get("title", ""), "citation": n.get("citation", ""),
             "source": n.get("source", ""), "article": n.get("article", "")}
            for n in normatives
        ]

        # Conflictos de contexto
        meanings = self.get_context_meanings(concept_id)
        if len(meanings) > 1:
            context["context_conflicts"] = [
                {"source_db": m.get("source_db", ""), "meaning": m.get("description", "")}
                for m in meanings
            ]

        # Comuestos y derivados
        context["composites"] = self.find_composites_of(concept_id)
        for source, _, data in self.graph.in_edges(concept_id, data=True):
            if data.get("type") == EdgeType.DERIVA_DE.value:
                n = self.graph.nodes[source]
                context["derived_from"].append({"id": source, "name": n.get("name", "?")})

        # Resumen de calidad
        scores = [f["quality_score"] for f in context["fields"] if f["quality_score"] > 0]
        context["quality_summary"] = {
            "avg_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
            "low_quality_count": sum(1 for s in scores if s < 0.4),
            "total_fields": len(context["fields"]),
        }

        return context

    # === PERSISTENCIA ===

    def find_all_classifiers(self) -> list[dict]:
        """Listar todos los clasificadores."""
        return [
            {"id": n, **d}
            for n, d in self.graph.nodes(data=True)
            if d.get("type") == NodeType.CLASSIFIER.value
        ]

    def save(self, path: str):
        """Persistir el grafo.

        Gap B: Si hay conexion PostgreSQL, hace write-through completo.
        Siempre guarda JSON local como backup/fallback.

        Optimistic concurrency: si el archivo JSON fue modificado externamente
        desde la ultima carga (mtime cambio), hace merge de los nodos nuevos
        antes de sobrescribir para evitar perdida de datos.
        """
        # Optimistic concurrency check para JSON local
        file_path = Path(path)
        if file_path.exists():
            current_mtime = file_path.stat().st_mtime
            if hasattr(self, "_loaded_mtime") and current_mtime != self._loaded_mtime:
                logger.warning(
                    "nomenclador.json fue modificado externamente (mtime %.2f -> %.2f). "
                    "Haciendo merge de nodos nuevos antes de guardar.",
                    self._loaded_mtime, current_mtime,
                )
                self._merge_from_disk(file_path)

        # JSON local (siempre, como backup)
        data = nx.node_link_data(self.graph, edges="links")
        data["_version"] = self.version
        data["_version_history"] = self.version_history
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Registrar mtime despues de guardar
        self._loaded_mtime = file_path.stat().st_mtime

        # PostgreSQL: registrar version (nodos/aristas ya fueron via write-through)
        self._db_save_version()

    def load(self, path: str):
        """Cargar el grafo.

        Gap B: Si hay conexion PostgreSQL con datos, hidrata desde ahi.
        Si no, fallback a JSON local.
        """
        # Intentar cargar desde PostgreSQL primero (Gap B)
        db_data = self._db_load_all()
        if db_data and db_data.get("nodes"):
            self.version = db_data.pop("_version", "1.0.0")
            self.graph = nx.DiGraph()
            for node_data in db_data["nodes"]:
                node_id = node_data.pop("id")
                self.graph.add_node(node_id, **node_data)
            for edge_data in db_data["links"]:
                source = edge_data.pop("source")
                target = edge_data.pop("target")
                self.graph.add_edge(source, target, **edge_data)
            self._loaded_mtime = Path(path).stat().st_mtime if Path(path).exists() else 0.0
            return

        # Fallback: cargar desde JSON local
        if not os.path.exists(path):
            self._loaded_mtime = 0.0
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.version = data.pop("_version", "1.0.0")
        self.version_history = data.pop("_version_history", [])
        edges_key = "links" if "links" in data else "edges"
        self.graph = nx.node_link_graph(data, edges=edges_key)
        self._loaded_mtime = Path(path).stat().st_mtime

        # Gap B: Si hay DB pero estaba vacia, sincronizar todo el JSON a PostgreSQL
        if self._db and self.graph.number_of_nodes() > 0:
            for node_id, node_data in self.graph.nodes(data=True):
                self._db_upsert_node(node_id, dict(node_data))
            for source, target, edge_data in self.graph.edges(data=True):
                edge_type = edge_data.get("type", "unknown")
                self._db_upsert_edge(source, target, edge_type, dict(edge_data))
            self._db_save_version()

    def _merge_from_disk(self, file_path: Path):
        """Merge optimista: cargar nodos/aristas del archivo en disco que no existen en memoria.

        Esto previene perdida de datos cuando dos procesos modifican el nomenclador.json
        concurrentemente. Solo agrega nodos/aristas que no estan en el grafo en memoria.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            edges_key = "links" if "links" in data else "edges"
            disk_graph = nx.node_link_graph(data, edges=edges_key)

            new_nodes = 0
            for node_id, node_data in disk_graph.nodes(data=True):
                if node_id not in self.graph:
                    self.graph.add_node(node_id, **node_data)
                    new_nodes += 1

            new_edges = 0
            for source, target, edge_data in disk_graph.edges(data=True):
                if not self.graph.has_edge(source, target):
                    self.graph.add_edge(source, target, **edge_data)
                    new_edges += 1

            if new_nodes or new_edges:
                logger.info("Merge desde disco: %d nodos nuevos, %d aristas nuevas", new_nodes, new_edges)
        except Exception as e:
            logger.warning("Merge desde disco fallo: %s — continuando con grafo en memoria", e)

    # === ESTADÍSTICAS ===

    def stats(self) -> dict:
        """Estadísticas del grafo."""
        counts = {}
        for _, d in self.graph.nodes(data=True):
            t = d.get("type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "by_type": counts,
            "version": self.version,
        }


# === Module-level cached loader ===

_GRAPH_CACHE: NomencladorGraph | None = None
_GRAPH_CACHE_MTIME: float = 0.0
_NOMENCLADOR_PATH = Path(__file__).parent.parent.parent / "nomenclador" / "nomenclador.json"


def load_graph_cached() -> NomencladorGraph:
    """Cargar el grafo con cache de lectura. Funcion publica para reuso.

    Invalida automaticamente el cache si el archivo nomenclador.json
    fue modificado en disco (por mtime).
    """
    global _GRAPH_CACHE, _GRAPH_CACHE_MTIME
    if _GRAPH_CACHE is not None:
        if _NOMENCLADOR_PATH.exists():
            current_mtime = _NOMENCLADOR_PATH.stat().st_mtime
            if current_mtime != _GRAPH_CACHE_MTIME:
                _GRAPH_CACHE = None
        if _GRAPH_CACHE is not None:
            return _GRAPH_CACHE
    g = NomencladorGraph()
    if _NOMENCLADOR_PATH.exists():
        g.load(str(_NOMENCLADOR_PATH))
        _GRAPH_CACHE_MTIME = _NOMENCLADOR_PATH.stat().st_mtime
    _GRAPH_CACHE = g
    return g


def clear_graph_cache():
    """Invalidar el cache del grafo. Llamar despues de modificar el nomenclador."""
    global _GRAPH_CACHE, _GRAPH_CACHE_MTIME
    _GRAPH_CACHE = None
    _GRAPH_CACHE_MTIME = 0.0