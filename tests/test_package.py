def test_exports():
    import pipecat_spatialreal
    from pipecat_spatialreal.proto.generated import avatar_pb2

    assert pipecat_spatialreal.__version__ == "0.2.0"
    m = avatar_pb2.AvatarMessage(tag="spatialreal-avatar")
    m.turn_start.turn_id = "r1"
    assert m.WhichOneof("payload") == "turn_start"
    assert avatar_pb2.State.STATUS_DEGRADED == 2
