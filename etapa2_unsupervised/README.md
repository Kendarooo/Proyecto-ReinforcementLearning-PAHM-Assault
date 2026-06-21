# etapa2_unsupervised

Esta carpeta se conserva como prototipo historico de Etapa 2. Implementa un
autoencoder simple sobre ventanas de `tau_w(t)` y pruebas heredadas del primer
contrato Etapa 1 -> Etapa 2.

La ruta vigente para la entrega es `pahm_stage2/`, porque cubre los requisitos
defendibles de la rubrica:

- extraccion de features de senales `tau_w(t)`;
- modelo GMM no supervisado;
- seleccion de complejidad mediante BIC;
- validacion sintetico-real con ARI/NMI;
- `WindSampler` como consumidor de la representacion aprendida.

Use esta carpeta solo para trazabilidad o comparacion historica. Para entrenar,
validar y documentar Etapa 2 en la entrega final, use:

```bash
.venv/bin/python -m pahm_stage2.validate_synthetic_real \
  --config configs/stage2_config.json

.venv/bin/python -m pahm_stage2.train_unsupervised \
  --config configs/stage2_config.json
```
