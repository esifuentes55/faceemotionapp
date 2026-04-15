import re
import sys
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


class EPGEdexExtractor:
    """Extractor de listas de precios EDEX con validaciones y manejo de errores."""

    def __init__(self, input_file_path: str, template_file_path: str) -> None:
        self.input_file = input_file_path
        self.template_file = template_file_path
        self.datos_acumulados: List[Dict[str, Any]] = []
        self.errores: List[str] = []

        self.input_data: Optional[pd.ExcelFile] = None
        self.template_data: Optional[pd.ExcelFile] = None
        self.sheet_cache: Dict[str, pd.DataFrame] = {}

        self.periodos_data = pd.DataFrame()
        self.carreras_data = pd.DataFrame()
        self.tipo_venta_data = pd.DataFrame()
        self.constantes: Dict[str, Any] = {}

    def _registrar_error(self, e: Exception) -> None:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        if exc_tb is None:
            self.errores.append(f"ERROR - Mensaje: {e}")
            return

        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        self.errores.append(
            f"ERROR - En el archivo: {file_name}, línea: {line_number}, Mensaje: {e}"
        )

    def _get_sheet_df(self, sheet_name: str) -> pd.DataFrame:
        if sheet_name not in self.sheet_cache:
            self.sheet_cache[sheet_name] = pd.read_excel(
                self.input_file, sheet_name=sheet_name, header=None
            )
        return self.sheet_cache[sheet_name]

    def get_constantes(self, sheet_name: str) -> Dict[str, Any]:
        """Extrae las variables constantes del template."""
        try:
            const_df = pd.read_excel(self.template_file, sheet_name=sheet_name)
            return dict(zip(const_df["Columna"], const_df["Valor"]))
        except Exception as e:
            self._registrar_error(e)
            return {}

    def load_files(self) -> None:
        try:
            self.input_data = pd.ExcelFile(self.input_file)
            self.template_data = pd.ExcelFile(self.template_file)

            self.periodos_data = pd.read_excel(self.template_file, sheet_name="Periodos")
            self.carreras_data = pd.read_excel(self.template_file, sheet_name="Carreras")
            self.tipo_venta_data = pd.read_excel(self.template_file, sheet_name="Tipo_Venta")
            self.constantes = self.get_constantes("variables_constantes")
        except Exception as e:
            self._registrar_error(e)

    @staticmethod
    def normalizar_texto(texto: Any) -> str:
        """Normaliza texto para comparación: mayúsculas, sin espacios extras, sin tildes."""
        if pd.isna(texto):
            return ""

        texto_norm = str(texto).strip().upper()
        reemplazos = {"Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U", "Ñ": "N"}
        for acento, sin_acento in reemplazos.items():
            texto_norm = texto_norm.replace(acento, sin_acento)

        return " ".join(texto_norm.split())

    @staticmethod
    def _to_date_string(value: Any, date_format: str = "%Y-%m-%d") -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.strftime(date_format)
        return str(value).strip()

    @staticmethod
    def _extract_year_from_text(texto: str) -> List[str]:
        """Extrae años válidos en formato 19xx/20xx y retorna strings únicos en orden."""
        years = re.findall(r"\b(19\d{2}|20\d{2})\b", texto)
        seen = set()
        ordered = []
        for y in years:
            if y not in seen:
                seen.add(y)
                ordered.append(y)
        return ordered

    def get_anioperio(self, sheet_name: str) -> str:
        """
        Columna ANIOPERIO.

        Nueva lógica:
        1) Busca años de 4 dígitos en TODO el texto de la hoja.
        2) Si encuentra más de uno, devuelve el más frecuente.
        3) Si no encuentra, usa fallback en celda C14 (índice [13,2]) con formato YYxxxx -> 20YY.
        """
        try:
            df = self._get_sheet_df(sheet_name)

            all_text = " ".join(df.fillna("").astype(str).values.ravel())
            years = self._extract_year_from_text(all_text)
            if years:
                counts = Counter(re.findall(r"\b(19\d{2}|20\d{2})\b", all_text))
                return counts.most_common(1)[0][0]

            valor = df.iloc[13, 2]
            if pd.notna(valor):
                valor_str = str(valor).strip()
                two_digit = re.search(r"\b(\d{2})", valor_str)
                if two_digit:
                    return f"20{two_digit.group(1)}"

            return ""
        except Exception as e:
            self._registrar_error(e)
            return ""

    def get_nperio(self, sheet_name: str) -> str:
        try:
            df = self._get_sheet_df(sheet_name)
            valor_str = str(df.iloc[13, 2]).strip()

            resultado = self.periodos_data[self.periodos_data["DPERIODO"] == valor_str]
            if not resultado.empty:
                return str(resultado.iloc[0]["NPERIO"])
            return ""
        except Exception as e:
            self._registrar_error(e)
            return ""

    def get_ccarre(self, sheet_name: str) -> str:
        try:
            df = self._get_sheet_df(sheet_name)
            dato1 = df.iloc[8, 2]
            dato2 = df.iloc[9, 2]

            if pd.isna(dato1) or pd.isna(dato2):
                return ""

            nombre_completo_norm = self.normalizar_texto(f"{dato1} {dato2}")

            for _, row in self.carreras_data.iterrows():
                if self.normalizar_texto(row["nombre"]) == nombre_completo_norm:
                    return str(row["CCARRE"])

            return ""
        except Exception as e:
            self._registrar_error(e)
            return ""

    def get_dcarre(self, sheet_name: str) -> str:
        try:
            df = self._get_sheet_df(sheet_name)
            dato1 = df.iloc[8, 2]
            dato2 = df.iloc[9, 2]

            if pd.isna(dato1) or pd.isna(dato2):
                return ""
            return f"{str(dato1).strip()} {str(dato2).strip()}"
        except Exception as e:
            self._registrar_error(e)
            return ""

    def get_finicio(self, sheet_name: str) -> str:
        try:
            df = self._get_sheet_df(sheet_name)
            return self._to_date_string(df.iloc[18, 2])
        except Exception as e:
            self._registrar_error(e)
            return ""

    def get_ffinal(self, sheet_name: str) -> str:
        try:
            df = self._get_sheet_df(sheet_name)
            return self._to_date_string(df.iloc[18, 3])
        except Exception as e:
            self._registrar_error(e)
            return ""

    def get_cmoneda(self, sheet_name: str) -> str:
        try:
            df = self._get_sheet_df(sheet_name)
            value = df.iloc[11, 2]
            return "" if pd.isna(value) else str(value).strip()
        except Exception as e:
            self._registrar_error(e)
            return ""

    def get_datos_venta_con_cuotas(self, sheet_name: str) -> List[Dict[str, Any]]:
        try:
            df = self._get_sheet_df(sheet_name)

            tipos_con_fechas = {34, 38, 45}

            fechas: List[Dict[str, Any]] = []
            for idx, fila_fecha in enumerate(range(21, 27)):
                emision = df.iloc[fila_fecha, 2]
                vencimiento = df.iloc[fila_fecha, 3]

                if pd.notna(emision):
                    fechas.append(
                        {
                            "nro_cuota": idx + 1,
                            "femision": self._to_date_string(emision, "%d/%m/%Y"),
                            "fvenci": self._to_date_string(vencimiento, "%d/%m/%Y") if pd.notna(vencimiento) else None,
                        }
                    )

            if not fechas:
                fechas.append({"nro_cuota": 1, "femision": None, "fvenci": None})

            tipos_venta: List[Dict[str, Any]] = []
            for fila in range(31, 44):
                descripcion = df.iloc[fila, 1]
                if pd.isna(descripcion):
                    continue

                descripcion_str = str(descripcion).strip()
                resultado = self.tipo_venta_data[self.tipo_venta_data["DESCRIPCION"] == descripcion_str]
                if resultado.empty:
                    continue

                ctipoventa = resultado.iloc[0]["CTIPOVENTA"]
                tiene_fechas = (fila + 1) in tipos_con_fechas

                if tiene_fechas:
                    for fecha in fechas:
                        tipos_venta.append(
                            {
                                "ctipoventa": ctipoventa,
                                "descripcion": descripcion_str,
                                "nro_cuota": fecha["nro_cuota"],
                                "femision": fecha["femision"],
                                "fvenci": fecha["fvenci"],
                            }
                        )
                else:
                    tipos_venta.append(
                        {
                            "ctipoventa": ctipoventa,
                            "descripcion": descripcion_str,
                            "nro_cuota": 1,
                            "femision": None,
                            "fvenci": None,
                        }
                    )

            return tipos_venta
        except Exception as e:
            self._registrar_error(e)
            return []

    def process_sheet(self, sheet_name: str) -> None:
        try:
            anioperio = self.get_anioperio(sheet_name)
            nperio = self.get_nperio(sheet_name)
            ccarre = self.get_ccarre(sheet_name)
            dcarre = self.get_dcarre(sheet_name)
            finicio = self.get_finicio(sheet_name)
            ffinal = self.get_ffinal(sheet_name)
            cmoneda = self.get_cmoneda(sheet_name)

            cempre = self.constantes.get("CEMPRE")
            cprogr = self.constantes.get("CPROGR")
            desde = self.constantes.get("DESDE")
            hasta = self.constantes.get("HASTA")
            tipobenef = self.constantes.get("TIPO_BENEF")
            concpbenef = self.constantes.get("CONCPBENEF")

            tipos_venta = self.get_datos_venta_con_cuotas(sheet_name)
            df = self._get_sheet_df(sheet_name)

            categorias: List[Dict[str, Any]] = []
            for col in range(2, 11):
                porcentaje = df.iloc[29, col]
                categoria = df.iloc[30, col]
                if pd.notna(categoria):
                    categorias.append(
                        {
                            "categoria": str(categoria).strip(),
                            "porcentaje": float(porcentaje) if pd.notna(porcentaje) else 0.0,
                        }
                    )

            for tv in tipos_venta:
                for cat in categorias:
                    self.datos_acumulados.append(
                        {
                            "CEMPRE": cempre,
                            "CPROGR": cprogr,
                            "ANIOPERIO": anioperio,
                            "NPERIO": nperio,
                            "CCARRE": ccarre,
                            "DCARRE": dcarre,
                            "LPRECIO": " ",
                            "FINICIO": finicio,
                            "FFINAL": ffinal,
                            "CTIPOVENTA": tv["ctipoventa"],
                            "NRO_CUOTA": tv["nro_cuota"],
                            "FEMISION": tv["femision"],
                            "FVENCI": tv["fvenci"],
                            "CMONEDA": cmoneda,
                            "CATEGORIA": cat["categoria"],
                            "CARTICULO": " ",
                            "NORDEN": " ",
                            "DESDE": desde,
                            "HASTA": hasta,
                            "TIPOBENEF": tipobenef,
                            "CONCPBENEF": concpbenef,
                            "PORCNTDSCTO": cat["porcentaje"],
                            "MTODSCTO": " ",
                            "FDESDE": " ",
                            "FHASTA": " ",
                            "ESTADO_FILA": " ",
                        }
                    )

        except Exception as e:
            self._registrar_error(e)

    def create_output_file(self, output_path: str) -> None:
        try:
            if not self.datos_acumulados:
                print("No hay datos acumulados para generar el archivo")
                return

            df_output = pd.DataFrame(self.datos_acumulados)
            columnas_orden = [
                "CEMPRE",
                "CPROGR",
                "ANIOPERIO",
                "NPERIO",
                "CCARRE",
                "DCARRE",
                "LPRECIO",
                "FINICIO",
                "FFINAL",
                "CTIPOVENTA",
                "TIPO_LISTA",
                "NRO_CUOTA",
                "FEMISION",
                "FVENCI",
                "CMONEDA",
                "CATEGORIA",
                "CARTICULO",
                "NORDEN",
                "DESDE",
                "HASTA",
                "PRECIO",
                "TIPOBENEF",
                "CONCPBENEF",
                "PORCNTDSCTO",
                "MTODSCTO",
                "FDESDE",
                "FHASTA",
                "CUPONDSCTO",
                "ESTADO_FILA",
            ]

            columnas_existentes = [c for c in columnas_orden if c in df_output.columns]
            df_output = df_output[columnas_existentes]

            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                df_output.to_excel(writer, sheet_name="Hoja1", index=False)
                if self.errores:
                    pd.DataFrame(self.errores, columns=["Error"]).to_excel(
                        writer, sheet_name="Errores", index=False
                    )

        except Exception as e:
            self._registrar_error(e)
            print(f"Error al crear archivo de salida: {e}")


def main(lst_path: List[str]) -> Optional[str]:
    """Función principal que ejecuta todo el proceso de extracción."""
    try:
        input_path, template_path, output_path = lst_path

        extractor = EPGEdexExtractor(
            input_file_path=input_path,
            template_file_path=template_path,
        )
        extractor.load_files()

        if extractor.input_data is None:
            return "ERROR - No se pudo cargar el archivo de entrada"

        for sheet in extractor.input_data.sheet_names:
            extractor.process_sheet(sheet)

        extractor.create_output_file(output_path)
        return None
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        if exc_tb is None:
            return f"ERROR - Mensaje: {e}"
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"ERROR - En el archivo: {file_name}, línea: {line_number}, Mensaje: {e}"


if __name__ == "__main__":
    lst_path = [
        r"C:\Users\spintom\Desktop\Lista Precios EPG\Input\Files_to_Process\EDEX\LP_Edex_INTAKE 2_2026_0804 (2).xlsx",
        r"C:\Users\spintom\Desktop\Lista Precios EPG\Input\Templates\Template_Config_EDEX.xlsx",
        r"C:\Users\spintom\Desktop\Lista Precios EPG\Ouput\salida_Edex.xlsx",
    ]
    main(lst_path)
