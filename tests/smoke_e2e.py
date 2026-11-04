"""Comprobación de punta a punta contra una API en marcha, con los modelos reales.

No es una evaluación ni una demo: usa entradas mínimas temporales para confirmar que las tres
modalidades recorren el pipeline real (Ollama + Qdrant). No mide la calidad de los resultados.

    python tests/smoke_e2e.py [--api http://localhost:8010] [--con-comentario]
"""
import argparse
import json
import sys
import uuid

import requests

CASOS = {
    "text_vs_text": {
        "refs": {"original.txt": "The council approved the new budget on 3 March.\n\nSpending on libraries rises by 2%."},
        "revisado": "The council approved the budget in March and library spending rises by 5%.",
    },
    "report_vs_news": {
        "refs": {"informe.md": "# Report\n\nUnemployment fell to 6.2% in March among adults aged 25 to 54.\n\nThe survey covers 12 regions."},
        "revisado": "Unemployment fell to 6.2% in March across the whole population.",
    },
    "news_vs_newsletter": {
        "refs": {"lluvia.txt": "Heavy rain closed the northern railway line on Monday.",
                 "puerto.txt": "The port reported record container traffic in the first quarter.",
                 "museo.txt": "The city museum will open a new wing dedicated to photography."},
        "revisado": "Rain closed the northern railway line.\n\nThe port had record traffic.\n\nThe museum opens a photography wing.",
    },
}


def post(api, modalidad, refs, revisado, **form):
    files = [("file_ref", (nombre, texto.encode("utf-8"))) for nombre, texto in refs.items()]
    return requests.post(f"{api}/validar", files=files, data={"modalidad": modalidad, "texto_query": revisado, **form})


def comprobar_modalidad(api, modalidad, caso, con_comentario):
    marca = uuid.uuid4().hex[:8]  # corpus nuevo en cada ejecución: la primera vez siempre indexa
    refs = {nombre: f"{texto}\n\nRun {marca}." for nombre, texto in caso["refs"].items()}
    metadatos = [{"title": nombre, "url": f"https://example.org/{nombre}"} for nombre in refs]
    form = {"ref_metadata": json.dumps(metadatos), "usar_agente": str(con_comentario).lower()}

    r = post(api, modalidad, refs, caso["revisado"], **form)
    assert r.status_code == 200, f"{modalidad}: {r.status_code} {r.text[:300]}"
    cuerpo = r.json()
    assert cuerpo["modalidad"] == modalidad, cuerpo["modalidad"]
    assert cuerpo["indice"]["reutilizado"] is False, f"{modalidad}: un corpus nuevo no debería reutilizar un índice"
    assert cuerpo["resultados"], f"{modalidad}: sin resultados"
    for res in cuerpo["resultados"]:
        assert res["chunk"] and res["chunk_location"]["location"]
        assert 1 <= len(res["relevant_chunks"]) <= 3
        assert len(res["sources"]) == len(res["scores"]) == len(res["relevant_chunks"])
        assert all(s["filename"] in refs and s["location"] and s["url"] for s in res["sources"])
        assert all(-1 <= s <= 1 for s in res["scores"]), res["scores"]
        if con_comentario:
            assert res.get("comentario") or res.get("comentario_error")
    if modalidad == "news_vs_newsletter":
        assert len(cuerpo["resultados"]) == 3, "cada párrafo de la newsletter debe ser un fragmento revisado"
        vistas = {s["filename"] for res in cuerpo["resultados"] for s in res["sources"]}
        assert vistas == set(refs), f"no se han recuperado todas las noticias: {vistas}"

    # El mismo corpus (texto y metadatos) se reutiliza; uno modificado no
    meta = {"ref_metadata": form["ref_metadata"]}
    igual = post(api, modalidad, refs, caso["revisado"], **meta).json()["indice"]
    assert igual["reutilizado"] is True, f"{modalidad}: el mismo corpus debería reutilizar su índice: {igual}"
    modificadas = {**refs, next(iter(refs)): f"{next(iter(refs.values()))} Changed."}
    distinto = post(api, modalidad, modificadas, caso["revisado"], **meta).json()["indice"]
    assert distinto["reutilizado"] is False, f"{modalidad}: un corpus modificado no debe reutilizar el índice: {distinto}"
    assert distinto["coleccion"] != cuerpo["indice"]["coleccion"], f"{modalidad}: misma colección con otro corpus"

    primera = cuerpo["resultados"][0]
    print(f"OK {modalidad}: {len(cuerpo['resultados'])} fragmento(s); mejor pasaje {primera['sources'][0]['filename']} "
          f"{primera['sources'][0]['location']} (similitud {primera['scores'][0]:.3f})")
    if con_comentario:
        print(f"   comentario: {(primera.get('comentario') or primera.get('comentario_error'))[:200]!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8010")
    parser.add_argument("--con-comentario", action="store_true", help="también genera el comentario con el LLM")
    args = parser.parse_args()

    for modalidad, caso in CASOS.items():
        comprobar_modalidad(args.api, modalidad, caso, args.con_comentario)

    r = requests.post(f"{args.api}/validar", data={"modalidad": "Normativo vs Funcional", "texto_ref": "a", "texto_query": "b"})
    assert r.status_code == 400, r.text
    print("OK modalidad desconocida rechazada con 400")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        sys.exit(f"FALLO: {e}")
