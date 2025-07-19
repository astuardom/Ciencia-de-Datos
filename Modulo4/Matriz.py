import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib.backends.backend_pdf import PdfPages

class AnalisisCientifico:
    def __init__(self, archivo_csv):
        self.df = pd.read_csv(archivo_csv)
        self.variables = ["temperatura", "humedad", "niveles_co2", "ph_suelo", "intensidad_luz"]
        self.modelo = None
        self.y_pred = None
        self.residuos = None
        self.metricas = {}
    
    def matriz_correlacion(self):
        return self.df[self.variables].corr()

    def entrenar_modelo(self):
        X = self.df[self.variables]
        y = self.df["crecimiento_planta"]
        self.modelo = LinearRegression()
        self.modelo.fit(X, y)
        self.y_pred = self.modelo.predict(X)
        self.residuos = y - self.y_pred

    def evaluar_modelo(self):
        y = self.df["crecimiento_planta"]
        self.metricas["MSE"] = mean_squared_error(y, self.y_pred)
        self.metricas["MAE"] = mean_absolute_error(y, self.y_pred)
        self.metricas["R2"] = r2_score(y, self.y_pred)

    def graficos(self):
        sns.set(style="whitegrid")

        plt.figure(figsize=(10, 8))
        sns.heatmap(self.matriz_correlacion(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Matriz de Correlación de Variables Ambientales")
        plt.savefig("salidas/matriz_correlacion.png")
        plt.close()

        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=self.y_pred, y=self.residuos)
        plt.axhline(0, color='red', linestyle='--')
        plt.title("Gráfico de Residuos")
        plt.xlabel("Valores Predichos")
        plt.ylabel("Residuos")
        plt.savefig("salidas/grafico_residuos.png")
        plt.close()

        coef = pd.Series(self.modelo.coef_, index=self.variables)
        plt.figure(figsize=(8, 5))
        coef.plot(kind='barh', color='skyblue')
        plt.title("Importancia de Variables (Coeficientes)")
        plt.savefig("salidas/grafico_coeficientes.png")
        plt.close()

        plt.figure(figsize=(8, 5))
        sns.boxplot(data=self.df, x="especie", y="crecimiento_planta", hue="especie", palette="pastel", legend=False)
        plt.title("Crecimiento de Plantas por Especie")
        plt.savefig("salidas/boxplot_especie.png")
        plt.close()

        correl = self.df.select_dtypes(include=[np.number]).corr()["crecimiento_planta"].sort_values(ascending=False)
        plt.figure(figsize=(5, 6))
        sns.heatmap(correl.to_frame(), annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title("Correlación con Crecimiento de Planta")
        plt.savefig("salidas/correlacion_crecimiento.png")
        plt.close()

    def exportar_pdf(self):
        with PdfPages("salidas/reporte_regresion_clases.pdf") as pdf:
            fig = plt.figure(figsize=(11, 8.5))
            plt.suptitle("Reporte Técnico de Correlación y Regresión Múltiple", fontsize=16, y=0.95)
            plt.text(0.1, 0.7, f"Registros analizados: {len(self.df)}", fontsize=12)
            plt.text(0.1, 0.6, "- Variables predictoras:\n  temperatura, humedad, niveles_co2, ph_suelo, intensidad_luz", fontsize=11)
            plt.text(0.1, 0.4, "Nota: Correlación alta ≠ causalidad. Puede haber relaciones indirectas u otras variables ocultas.", fontsize=11)
            plt.axis("off")
            pdf.savefig(fig)
            plt.close()

            # Matriz de correlación
            img = plt.imread("salidas/matriz_correlacion.png")
            fig = plt.figure(figsize=(11, 8.5))
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig(fig)
            plt.close()

            # Métricas
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axis("off")
            for i, (k, v) in enumerate(self.metricas.items()):
                ax.text(0.1, 0.8 - i * 0.2, f"{k}: {v:.3f}", fontsize=14)
            ax.set_title("Evaluación del Modelo", fontsize=16, pad=20)
            pdf.savefig(fig)
            plt.close()

            # Gráficos
            for archivo in ["grafico_residuos.png", "grafico_coeficientes.png", "boxplot_especie.png", "correlacion_crecimiento.png"]:
                img = plt.imread(f"salidas/{archivo}")
                fig = plt.figure(figsize=(11, 8.5))
                plt.imshow(img)
                plt.axis("off")
                pdf.savefig(fig)
                plt.close()

# Ejecución
if __name__ == "__main__":
    analisis = AnalisisCientifico("entrada/science_data.csv")
    analisis.entrenar_modelo()
    analisis.evaluar_modelo()
    analisis.graficos()
    analisis.exportar_pdf()
    print("✅ Reporte generado: salidas/reporte_regresion_clases.pdf")
