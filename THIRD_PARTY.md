# Third-party components

The knowledge attention implementations in `src/atlaskv` and `src/kblam` derive from:

- [AtlasKV](https://github.com/HKUST-KnowComp/AtlasKV), revision `61e3abc1fb09fbfb426cfc847c4c26c5e94bcaeb`.
- [KBLaM](https://github.com/microsoft/KBLaM), revision `69683ad096e022b58854fd88c0c58968c7d8c470`.

The packaged versions include the attention callbacks and Llama compatibility changes used in the reported experiments. Their original license notices are retained in `LICENSES/` and source headers.

All four methods use the same frozen Llama-3.1-8B-Instruct language backbone and precomputed all-MiniLM-L6-v2 record embeddings. The SR-KI implementation uses supervised attention in this common interface: the layer-15 knowledge Query and Key branches are adapted while the Value branch stays frozen. It is the common-backbone implementation evaluated in the paper.

The base language checkpoint is downloaded from the original AtlasKV model distribution. It retains its original model terms. The included trained parameters contain the fitted knowledge interface, rather than the frozen language backbone. Dataset terms follow the original CC, Enron, and Pes2O sources used by AtlasKV.

Built with Llama. Llama 3.1 is licensed under the Llama 3.1 Community License; the notice is included in `LICENSES/Llama-3.1.txt`. The project MIT license applies to the reproduction code, subject to the retained third-party notices.
