---
title: "Keep Secrets Out of Your .env File"
slug: "keep-secrets-out-of-your-env-file"
date: 2026-06-10
tags: [cleancode, security]
mathjax: false
---

Recently there was a slew of supply chain attacks on several Python packages.
One of them, named [Shai-Hulud](https://devops-daily.com/posts/shai-hulud-hades-pypi-wave-june-2026), was especially impressive because you only had to install the compromised package, no import needed.
It then steals any credentials and secrets it can find on your machine, including ones in `.env` files.

My main question was: do people still keep secrets in environment variables?
API keys just lying around in plain text `.env` files on your hard drive sound like a terrible idea.
Especially when the alternative is so simple and elegant.
This is why I'm sharing my favorite pattern for fetching secrets in Python: _the secret resolver_.
It reads your secrets during application runtime from the secret provider of your choice, be it Azure Key Vault, KeePass, or 1Password.

## OmegaConf

This pattern is compatible with most configuration management packages[^1], but it's most easily explained with OmegaConf.
A resolver in OmegaConf reacts to configuration values that look like this: `"${resolver_name:key}"`.
The `resolver_name` is registered with OmegaConf, which then calls the resolver with `key` to get the value.
A good example is the built-in resolver `oc.env`, which reads environment variables.
The configuration value `"${oc.env:API_KEY}"` is resolved to the environment variable `API_KEY`.

A basic secret resolver using Azure Key Vault as the secret provider has only a few lines of code:

```python
from azure.keyvault.secrets import SecretClient
from omegaconf import OmegaConf


class AzureKeyVaultResolver:
    def __init__(self, client: SecretClient):
        self.client = client

    def __call__(self, name: str) -> str:
        return str(self.client.get_secret(name).value)


def register_secret_resolver(client: SecretClient) -> None:
    OmegaConf.register_new_resolver(
        "secret",
        AzureKeyVaultResolver(client),
        replace=True
    )

```

Before reading our config, we simply call the registration function with our secret client:

```python
credential = DefaultAzureCredential()
secret_client = SecretClient(vault_url="<your_vault>", credential=credential)
register_secret_resolver(secret_client)
conf = OmegaConf.create(
    {"not_a_secret": "${oc.env:NOT_A_SECRET}", "api_key": "${secret:API_KEY}"}
)
```

Accessing `conf.not_a_secret` will read the value from the environment variable `NOT_A_SECRET`.
Now, if you access `conf.api_key`, the resolver will fetch it for you from your key vault.
The value is cached in memory for subsequent access.
You can even use nested resolvers to read the name of the secret from an environment variable if that is something you want:

```python
conf = OmegaConf.create({"api_key": "${secret:${oc.env:API_KEY}}"})
```

## Pydantic-Settings

Okay, so you don't use OmegaConf, but pydantic-settings.
Can we use the secret resolver pattern as well?
Of course, by leveraging `Annotated` type hints:

```python
from dataclasses import dataclass
from typing import Any, Dict, Type

from azure.keyvault.secrets import SecretClient
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource


@dataclass
class Secret:
    name: str


class AzureSecretSource(PydanticBaseSettingsSource):
    def __init__(
        self,
        settings_cls: Type[BaseSettings],
        client: SecretClient,
    ):
        super().__init__(settings_cls)
        self.client = client

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        marker = next((m for m in field.metadata if isinstance(m, Secret)), None)

        if marker and marker.name:
            secret = self.client.get_secret(marker.name)
            return str(secret.value), field_name, False

        return None, field_name, False

    def __call__(self) -> Dict[str, Any]:
        data = {}
        for field_name, field in self.settings_cls.model_fields.items():
            value, _, _ = self.get_field_value(field, field_name)
            if value is not None:
                data[field_name] = value

        return data


def resolve_secrets_in_sources(client: SecretClient):

    def wrapper(
        settings_cls: Type[BaseSettings], **kwargs: PydanticBaseSettingsSource
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return AzureSecretSource(settings_cls, client), *kwargs.values()

    return wrapper

```

This version is a little bit more verbose than OmegaConf's, but it lets us use type-safe Pydantic models for configuration:

```python
from typing import Annotated

from pydantic_settings import BaseSettings

class Config(BaseSettings):
    not_a_secret: str
    api_key: Annotated[str, Secret]
    
    @classmethod
    def settings_customise_sources(cls, settings_cls, **kwargs):
        credential = DefaultAzureCredential()
        secret_client = SecretClient(
            vault_url="<your_vault>", credential=credential
        )
        
        return resolve_secrets_in_sources(secret_client)(settings_cls, **kwargs)


config = Config()
```

By default, pydantic-settings will populate attributes like `not_a_secret` from environment variables of the same name.
The `settings_customise_sources` class method handles the resolver registration.
Accessing `config.api_key` now works similarly to OmegaConf, but instead of fetching secrets lazily, all of them are resolved on construction of `config`.

## Conclusion

Don't get duped by secret-stealing supply chain attacks.
Use the _secret resolver_ pattern.
Both examples were generated using a coding agent in under half an hour[^2].
I am confident your coding agent of choice can build something similar for whatever configuration management package you use.

[^1]: If you're still reading from `os.environ` directly, please stop.
[^2]: Including tests to verify they work.