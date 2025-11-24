🌍 [Read in English](README.md)

# Skill Bridge Api Gen Ai

Web Api rest para o projeto skill bridge, para disponibilizar um modelo criado em uma api.

## Rotas

(Geração)

- `GET api/generate` - Use the model.

(Documentação)

- `GET api/docs` - The swagger docs

(Descrição)

- `GET api` - The description of the api

## Passos para rodar

1. Clone o repositório:

```bash
git clone https://github.com/felipeclarindo/skill-bridge-api-gen-ai-python.git
```

2. Entre no repositório:

```bash
cd skill-bridge-api-gen-ai-python
```

3. Execute as celular jupyter no arquivo `file_turing_generative_ai.ipynb` para treinar e gerar o modelo localizado em: `src/notebooks`.

4. Execute a api:

```bash
python src/api/api.py
```

5. A API está disponível em:

- <http://localhost:8000/api>

## Equipe

- RM: 554547 -> Felipe Gabriel Lopes Pinheiro Clarindo
- RM: 558482 -> Humberto de Souza Silva
- RM: 554914 -> Vinicius Beda de Oliveira

## Contribuição

Contribuições são bem-vindas! Se você tiver sugestões de melhorias, sinta-se à vontade para abrir uma issue ou enviar um pull request.

## Licença

Este projeto está licenciado sob a [GNU Affero License](https://www.gnu.org/licenses/agpl-3.0.html).
