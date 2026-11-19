"""E10, criterios de aceptación (1)–(4): conformidad de las primitivas y del diseño del perfil S2.

  (1) C1–C5: vectores públicos (RFC 5869 HKDF, casos de prueba de GCM de McGrew–Viega para AES-256 —los
      mismos que usa la validación de NIST—, RFC 5903 ECDH P-256, RFC 6979 ECDSA P-256/SHA-256) y vinculación
      del transcript: cambiar cualquier byte invalida la firma.
  (2) Ninguna clave se repite en 10⁶ derivaciones.
  (3) Perder cualquier subconjunto de mensajes NO impide descifrar los siguientes (resuelve R-01).
  (4) Replay y duplicados rechazados al 100 %.
"""
from __future__ import annotations

import random

import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from messaging.security import envelope
from messaging.security.handshake import (auth_payload, build_transcript, derive_session, message_key,
                                          message_nonce)
from messaging.security.primitives import (CryptoError, aead_open, aead_seal, decode_point, ecdh, encode_point,
                                           hkdf_expand, hkdf_extract, sign, verify)
from messaging.security.replay import ReplayWindow

H = bytes.fromhex


# ---------------------------------------------------------------------------- C1: HKDF (RFC 5869, apéndice A)
@pytest.mark.parametrize("ikm,salt,info,length,prk,okm", [
    # Test Case 1
    ("0b" * 22, "000102030405060708090a0b0c", "f0f1f2f3f4f5f6f7f8f9", 42,
     "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5",
     "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"),
    # Test Case 2 (entradas largas)
    ("".join(f"{i:02x}" for i in range(0x00, 0x50)), "".join(f"{i:02x}" for i in range(0x60, 0xb0)),
     "".join(f"{i:02x}" for i in range(0xb0, 0x100)), 82,
     "06a6b88c5853361a06104c9ceb35b45cef760014904671014a193f40c15fc244",
     "b11e398dc80327a1c8e7f78c596a49344f012eda2d4efad8a050cc4c19afa97c59045a99cac7827271cb41c65e590e09"
     "da3275600c2f09b8367793a9aca3db71cc30c58179ec3e87c14c01d5c1f3434f1d87"),
    # Test Case 3 (sal e info vacías)
    ("0b" * 22, "", "", 42,
     "19ef24a32c717b167f33a91d6f648bdf96596776afdb6377ac434c1c293ccb04",
     "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d9d201395faa4b61a96c8"),
])
def test_c1_hkdf_rfc5869(ikm, salt, info, length, prk, okm):
    got_prk = hkdf_extract(H(salt), H(ikm))
    assert got_prk.hex() == prk
    assert hkdf_expand(got_prk, H(info), length).hex() == okm


# ---------------------------------------------------------------------------- C2: AES-256-GCM
K15 = "feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308"
P15 = ("d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525"
       "b16aedf5aa0de657ba637b391aafd255")
C15 = ("522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa8cb08e48590dbb3da7b08b1056828838"
       "c5f61e6393ba7a0abcc9f662898015ad")


@pytest.mark.parametrize("key,iv,pt,aad,ct,tag", [
    ("00" * 32, "00" * 12, "", "", "", "530f8afbc74536b9a963b4f1c4cb738b"),                          # TC13
    ("00" * 32, "00" * 12, "00" * 16, "", "cea7403d4d606b6e074ec5d3baf39d18",
     "d0d1c8a799996bf0265b98b5d48ab919"),                                                          # TC14
    (K15, "cafebabefacedbaddecaf888", P15, "", C15, "b094dac5d93471bdec1a502270e3cc6c"),           # TC15
    (K15, "cafebabefacedbaddecaf888", P15[:120], "feedfacedeadbeeffeedfacedeadbeefabaddad2", C15[:120],
     "76fc6ece0f4e1768cddf8853bb2d551b"),                                                          # TC16
])
def test_c2_aes256_gcm_vectors(key, iv, pt, aad, ct, tag):
    out = aead_seal(H(key), H(iv), H(pt), H(aad))
    assert out.hex() == ct + tag
    assert aead_open(H(key), H(iv), out, H(aad)) == H(pt)
    if out:
        bad = bytearray(out)
        bad[-1] ^= 1
        with pytest.raises(CryptoError):
            aead_open(H(key), H(iv), bytes(bad), H(aad))


# ---------------------------------------------------------------------------- C3: ECDH P-256 (RFC 5903 §8.1)
def test_c3_ecdh_p256_rfc5903():
    i = ec.derive_private_key(int("C88F01F510D9AC3F70A292DAA2316DE544E9AAB8AFE84049C62A9C57862D1433", 16),
                              ec.SECP256R1())
    r = ec.derive_private_key(int("C6EF9C5D78AE012A011164ACB397CE2088685D8F06BF9BE0B283AB46476BEE53", 16),
                              ec.SECP256R1())
    gi = encode_point(i.public_key()).hex().upper()
    gr = encode_point(r.public_key()).hex().upper()
    assert gi == "04" + "DAD0B65394221CF9B051E1FECA5787D098DFE637FC90B9EF945D0C3772581180" \
                        "5271A0461CDB8252D61F1C456FA3E59AB1F45B33ACCF5F58389E0577B8990BB3"
    assert gr == "04" + "D12DFB5289C8D4F81208B70270398C342296970A0BCCB74C736FC7554494BF63" \
                        "56FBF3CA366CC23E8157854C13C58D6AAC23F046ADA30F8353E74F33039872AB"
    shared = "D6840F6B42F6EDAFD13116E0E12565202FEF8E9ECE7DCE03812464D04B9442DE"
    assert ecdh(i, decode_point(H(gr))).hex().upper() == shared
    assert ecdh(r, decode_point(H(gi))).hex().upper() == shared


def test_c3_invalid_point_rejected():
    good = bytearray(encode_point(ec.generate_private_key(ec.SECP256R1()).public_key()))
    good[-1] ^= 1                                                    # punto fuera de la curva
    with pytest.raises(CryptoError):
        decode_point(bytes(good))


# ---------------------------------------------------------------------------- C4: ECDSA P-256 (RFC 6979 A.2.5)
RFC6979_X = "C9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721"


@pytest.mark.parametrize("msg,r,s", [
    (b"sample", "EFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716",
     "F7CB1C942D657C41D436C7A1B6E29F65F3E900DBB9AFF4064DC4AB2F843ACDA8"),
    (b"test", "F1ABB023518351CD71D881567B1EA663ED3EFCF6C5132B354F28D3B0B7D38367",
     "019F4113742A2B14BD25926B49C649155F267E60D3814B4C0CC84250E46F0083"),
])
def test_c4_ecdsa_p256_rfc6979(msg, r, s):
    sk = ec.derive_private_key(int(RFC6979_X, 16), ec.SECP256R1())
    pub = encode_point(sk.public_key()).hex().upper()
    assert pub == "04" + "60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6" \
                         "7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299"
    sig = H(r + s)
    verify(sk.public_key(), sig, msg)                                # vector conocido (k determinista)
    with pytest.raises(CryptoError):
        verify(sk.public_key(), sig, msg + b"!")
    verify(sk.public_key(), sign(sk, msg), msg)                      # nuestra firma (k aleatorio) r‖s


# ---------------------------------------------------------------------------- C5: vinculación del transcript
def _hello(node_id: str) -> dict:
    sk = ec.generate_private_key(ec.SECP256R1())
    return {"node_id": node_id, "kid": "ab" * 16, "nonce": bytes(range(32)) if node_id == "CN1" else bytes(32),
            "epk": encode_point(sk.public_key())}


def test_c5_any_transcript_byte_change_invalidates_signature():
    a, b = _hello("CN1"), _hello("infra")
    t = build_transcript("exp-1", a, b)
    assert t == build_transcript("exp-1", b, a)                      # independiente del orden de llegada
    sk = ec.generate_private_key(ec.SECP256R1())
    sig = sign(sk, auth_payload(t, "CN1"))
    for i in range(len(t)):                                          # TODOS los bytes
        mod = bytearray(t)
        mod[i] ^= 0x01
        with pytest.raises(CryptoError):
            verify(sk.public_key(), sig, auth_payload(bytes(mod), "CN1"))
    with pytest.raises(CryptoError):                                 # reflejar la firma como la del otro extremo
        verify(sk.public_key(), sig, auth_payload(t, "infra"))


def test_c5_transcript_binds_identities_experiment_and_keys():
    a, b = _hello("CN1"), _hello("infra")
    base = build_transcript("exp-1", a, b)
    variants = [build_transcript("exp-2", a, b), build_transcript("exp-1", {**a, "node_id": "CN2"}, b),
                build_transcript("exp-1", {**a, "kid": "cd" * 16}, b),
                build_transcript("exp-1", {**a, "epk": _hello("CN1")["epk"]}, b),
                build_transcript("exp-1", a, {**b, "nonce": bytes([1]) * 32})]
    assert all(v != base for v in variants)


# ---------------------------------------------------------------------------- sesiones de prueba
def session_pair(epoch_seed: int = 0):
    t = build_transcript("exp-1", _hello("CN1"), _hello("infra")) + epoch_seed.to_bytes(4, "big")
    shared, nonces = bytes(range(32)), bytes(64)
    return derive_session("CN1", "infra", t, shared, nonces), derive_session("infra", "CN1", t, shared, nonces)


def test_sessions_agree_and_directions_differ():
    a, b = session_pair()
    assert a.epoch == b.epoch and a.k_send == b.k_recv and a.k_recv == b.k_send and a.k_send != a.k_recv


# ---------------------------------------------------------------------------- (2) unicidad de claves
@pytest.mark.slow
def test_no_key_repeats_in_one_million_derivations():
    a, _ = session_pair()
    keys = set()
    n = 0
    for k_dir in (a.k_send, a.k_recv):                               # dos direcciones × 500 000 seq
        for seq in range(1, 500_001):
            keys.add(message_key(k_dir, a.epoch, seq))
            n += 1
    assert n == 1_000_000 and len(keys) == n
    assert len({message_nonce(a.epoch, s) for s in range(1, 10_001)}) == 10_000


# ---------------------------------------------------------------------------- (3) pérdidas
def test_losing_any_subset_does_not_prevent_decrypting_the_rest():
    rng = random.Random(42)
    for trial in range(20):
        tx, rx = session_pair(trial)
        frames = [envelope.seal(tx, f"m{i}".encode() * 50, "UPDATE", "exp-1") for i in range(200)]
        p_loss = rng.choice([0.1, 0.5, 0.9])
        kept = [i for i in range(200) if rng.random() > p_loss]
        if trial == 0:
            kept = [199]                                             # solo el último sobrevive
        for i in kept:
            outer = envelope.parse(frames[i], 1 << 20)
            assert rx.window.check(outer.seq) is None
            assert envelope.open_(rx, outer) == f"m{i}".encode() * 50
            rx.window.mark(outer.seq)


def test_tampering_with_outer_header_or_ciphertext_fails():
    tx, rx = session_pair()
    frame = bytearray(envelope.seal(tx, b"payload" * 10, "UPDATE", "exp-1"))
    for pos in (6, 20, len(frame) - 30, len(frame) - 1):            # cabecera externa (AAD), cifrado, etiqueta
        bad = bytearray(frame)
        bad[pos] ^= 0x01
        try:
            outer = envelope.parse(bytes(bad), 1 << 20)
        except envelope.EnvelopeError:
            continue                                                 # la estructura ya no es válida
        with pytest.raises(CryptoError):
            envelope.open_(rx, outer)


# ---------------------------------------------------------------------------- (4) replay
def test_replay_and_duplicates_rejected_100_percent():
    tx, rx = session_pair()
    frames = [envelope.seal(tx, b"x", "MODEL", "exp-1") for _ in range(300)]
    accepted = 0
    for f in frames:
        o = envelope.parse(f, 1 << 20)
        assert rx.window.check(o.seq) is None
        envelope.open_(rx, o)
        rx.window.mark(o.seq)
        accepted += 1
    rejected = 0
    for f in frames + frames[::-1]:                                  # todos repetidos, en orden y al revés
        o = envelope.parse(f, 1 << 20)
        if rx.window.check(o.seq) is not None:
            rejected += 1
    assert accepted == 300 and rejected == 600


def test_replay_window_tolerates_gaps_and_reordering():
    w = ReplayWindow(8)
    for s in (1, 2, 5, 4, 9, 3):                                     # huecos y reordenación dentro de W
        assert w.check(s) is None
        w.mark(s)
    assert w.check(5) is not None and w.check(9) is not None        # duplicados
    assert w.check(1) is not None                                    # 9 - 1 = 8 ≥ W: fuera de la ventana
    assert w.check(6) is None and w.check(100) is None
